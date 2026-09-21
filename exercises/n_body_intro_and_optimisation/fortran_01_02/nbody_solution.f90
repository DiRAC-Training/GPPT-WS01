module nbody_simulation
    implicit none

#ifdef SINGLE_PRECISION
    integer, parameter :: wp = selected_real_kind(6, 37) ! single precision
#else
    integer, parameter :: wp = kind(1.0d0) ! double precision
#endif
    real(wp), parameter :: PI = 3.14159265358979323846_wp
    real(wp), parameter :: N_YEARS = 0.1_wp
    integer :: file_unit, ios

contains

    subroutine random_numbers(num, a, b, res)
        integer, intent(in) :: num
        real(wp), intent(in) :: a, b
        real(wp), intent(out) :: res(num)
        real(wp) :: temp(num)

        call random_number(temp)
        res = temp * (b - a) + a
    end subroutine random_numbers

    subroutine calc_stable_orbit(r, theta, pos, vel)
        real(wp), intent(in) :: r(:)
        real(wp), intent(in) :: theta(:)
        real(wp), intent(out) :: pos(:,:)
        real(wp), intent(out) :: vel(:,:)

        integer :: n
        real(wp), allocatable :: v_mag(:)

        n = size(r)
        allocate(v_mag(n))

        v_mag = 1.0_wp / sqrt(r)

        pos(:,1) = r * sin(theta)
        pos(:,2) = r * cos(theta)

        vel(:,1) = -v_mag * cos(theta)
        vel(:,2) =  v_mag * sin(theta)

        deallocate(v_mag)
    end subroutine calc_stable_orbit

    subroutine generate_random_star_system(num, pos, vel, mass, min_radius, max_radius, min_mass, max_mass)
        integer, intent(in) :: num
        real(wp), intent(out) :: pos(num, 2)
        real(wp), intent(out) :: vel(num, 2)
        real(wp), intent(out) :: mass(num)
        real(wp), intent(in), optional :: min_radius, max_radius, min_mass, max_mass

        real(wp) :: r_min, r_max, m_min, m_max
        real(wp), allocatable :: r(:), theta(:)

        r_min = 0.4_wp; if (present(min_radius)) r_min = min_radius
        r_max = 20.0_wp; if (present(max_radius)) r_max = max_radius
        m_min = 1.0_wp/6000000.0_wp; if (present(min_mass)) m_min = min_mass
        m_max = 1.0_wp/1000.0_wp; if (present(max_mass)) m_max = max_mass

        allocate(r(num), theta(num))

        call random_numbers(num, r_min, r_max, r)
        call random_numbers(num, m_min, m_max, mass)
        call random_numbers(num, 0.0_wp, PI, theta)

        call calc_stable_orbit(r, theta, pos, vel)

        ! Add central star
        pos(1,:) = 0.0_wp
        vel(1,:) = 0.0_wp
        mass(1) = 1.0_wp

        deallocate(r, theta)
    end subroutine generate_random_star_system

#ifdef TILED
    subroutine calc_acc_tiled(acc, pos, mass)
        use omp_lib, only: omp_get_thread_num
        real(wp), intent(out) :: acc(:,:)
        real(wp), intent(in) :: pos(:,:)
        real(wp), intent(in) :: mass(:)

        integer, parameter :: TILE = 128
        integer :: n, num_teams_needed
        integer :: team_id, tid, i, t, tile_i, j
        real(wp) :: epsilon, dx, dy, dist_sq, inv_dist_cube, ax, ay

        real(wp) :: pos_s(TILE, 2)
        real(wp) :: ??? ! TODO create a shared array for the mass tile

        n = size(pos, 1)
        epsilon = 1.1_wp * (real(n, wp)**(-0.48_wp))
        num_teams_needed = ??? ! TODO how many tiles do we need to cover n?

        ! TODO these omp directives are out of order! Fix them
        !$omp distribute private(pos_s, mass_s)
        !$omp target teams num_teams(num_teams_needed) thread_limit(TILE)
        !$omp parallel private(tid, i, ax, ay, t, tile_i, j, dx, dy, dist_sq, inv_dist_cube)
        do team_id = 0, num_teams_needed - 1
            tid = omp_get_thread_num()
            i = ??? ! TODO calculate the global index for this thread
            ax = 0.0_wp
            ay = 0.0_wp

            do t = 0, num_teams_needed - 1
                ! TODO do we need a barrier here?

                tile_i = ??? ! TODO similar to i, calc global index for this thread's tile position
                ! Load the tile
                if (tile_i <= n) then
                    pos_s(tid+1, 1) = pos(tile_i, 1)
                    pos_s(tid+1, 2) = pos(tile_i, 2)
                    mass_s(tid+1)   = mass(tile_i)
                else
                    pos_s(tid+1, 1) = 0.0_wp
                    pos_s(tid+1, 2) = 0.0_wp
                    mass_s(tid+1)   = 0.0_wp
                end if
                ! TODO do we need a barrier here?

                ! Calc acc using tile data
                if (i <= n) then
                    do j = 1, TILE
                        dx = pos_s(j,1) - pos(i,1)
                        dy = ??? ! TODO
                        dist_sq = dx**2 + dy**2 + epsilon**2
                        inv_dist_cube = 1.0_wp / (dist_sq * sqrt(dist_sq))
                        ax = ax + dx * mass??? * inv_dist_cube ! TODO how to access mass within the tile?
                        ay = ay + dy * mass??? * inv_dist_cube
                    end do
                end if
                ! TODO do we need a barrier here?
            end do

            ! TODO I'm worried this will access out-of-bounds. Should there be a guard?
            acc(i,1) = ax
            acc(i,2) = ay
            !$omp end parallel
        end do
        !$omp end distribute
        !$omp end target teams
    end subroutine calc_acc_tiled
#endif ! TILED


    ! Computes the gravitational acceleration on every particle from every
    ! other particle: O(n^2) work, and the hottest part of the simulation by
    ! far. This is the first thing to move onto the GPU.
    subroutine calc_acc(acc, pos, mass)
        real(wp), intent(inout) :: acc(:,:)
        real(wp), intent(in) :: pos(:,:)
        real(wp), intent(in) :: mass(:)

        integer :: i, j, n
        real(wp) :: epsilon, dx, dy, dist_sq, inv_dist_cube

#ifdef TILED
        call calc_acc_tiled(acc, pos, mass)
        return
#endif

        n = size(pos, 1)
        epsilon = 1.1_wp * (real(n, wp)**(-0.48_wp))

        !$omp target teams distribute parallel do
        do i = 1, n
          acc(i,1) = 0.0_wp
          acc(i,2) = 0.0_wp
        enddo
        !$omp end target teams distribute parallel do
        
        !$omp target teams distribute parallel do
        do i = 1, n
            do j = 1, n
                dx = pos(j,1) - pos(i,1)
                dy = pos(j,2) - pos(i,2)
                dist_sq = dx**2 + dy**2 + epsilon**2
                ! inv_dist_cube = 1.0_wp / (dist_sq**1.5_wp)
                inv_dist_cube = 1.0_wp / (dist_sq * sqrt(dist_sq))
                acc(i,1) = acc(i,1) + dx * mass(j) * inv_dist_cube
                acc(i,2) = acc(i,2) + dy * mass(j) * inv_dist_cube
            end do
        end do
        !$omp end target teams distribute parallel do
    end subroutine calc_acc

    ! Velocity-Verlet position update. Runs once per particle per step, so it
    ! is much cheaper than calc_acc, but it still touches every array once a
    ! step and is worth offloading so the data doesn't have to travel back to
    ! the host in between.
    subroutine advance_pos(acc, pos, pos_prev, pos_temp, dt)
        real(wp), intent(in) :: acc(:,:)
        real(wp), intent(inout) :: pos(:,:)
        real(wp), intent(inout) :: pos_prev(:,:)
        real(wp), intent(out) :: pos_temp(:,:)
        real(wp), intent(in) :: dt
        integer :: i, n

        n = size(pos, 1)

        !$omp target teams distribute parallel do
        do i=1,n
            pos_temp(i,1) = pos(i,1)
            pos_temp(i,2) = pos(i,2)
            pos(i,1) = 2.0_wp * pos(i,1) - pos_prev(i,1) + acc(i,1) * dt**2
            pos(i,2) = 2.0_wp * pos(i,2) - pos_prev(i,2) + acc(i,2) * dt**2
            pos_prev(i,1) = pos_temp(i,1)
            pos_prev(i,2) = pos_temp(i,2)
        end do
        !$omp end target teams distribute parallel do
    end subroutine advance_pos

    subroutine run_sim(n_particles, n_steps)
        integer, intent(in) :: n_particles
        integer, intent(in) :: n_steps

        integer :: current_step = 0
        integer :: print_every
        real(wp) :: completion_time
        real(wp) :: dt

        real(wp), allocatable :: pos(:,:), vel(:,:), mass(:)
        real(wp), allocatable :: acc(:,:), pos_temp(:,:), pos_prev(:,:)
        integer :: n, count_rate, count_start, count_end
        integer :: i

        print *, "Running with ", n_particles, " particles"
        n = n_particles

        dt = 0.01_wp
        print_every = max(floor(real(n_steps) / 10), 1) ! Print every N steps

        allocate(pos(n, 2), vel(n, 2), mass(n))
        allocate(acc(n, 2), pos_temp(n, 2), pos_prev(n, 2))

        call generate_random_star_system(n, pos, vel, mass)

        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc(acc, pos, mass)
        !$omp end target data

        pos_prev = pos - vel * dt - 0.5_wp * acc * dt**2

        !$omp target data map(tofrom: pos) map(to:mass, pos_prev) map(alloc: acc, pos_temp)
        call system_clock(count_start, count_rate)
        
        do while (current_step < n_steps)
            call calc_acc(acc, pos, mass)
            call advance_pos(acc, pos, pos_prev, pos_temp, dt)
            current_step = current_step + 1
            if (mod(current_step, print_every) .eq. 0) then
              print *, "Complete: ", real(current_step) / real(n_steps) * 100
            end if
        end do
        call system_clock(count_end)
        !$omp end target data

        if (ios == 0) then
            do i = 1, n
                write(file_unit, '(F12.6, A, F12.6)') pos(i, 1), ',', pos(i, 2)
            end do
        end if

        completion_time = real(count_end - count_start, wp) / real(count_rate, wp)

        print '(A, F10.4, A)', "Time to complete: ", completion_time, " s"
        print '(A, F10.4, A)', "Mean time per step: ", completion_time / n_steps, " s"

        deallocate(pos, vel, mass, acc, pos_temp, pos_prev)
    end subroutine run_sim

    subroutine assert_almost_equal(a, b, label)
        real(wp), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: label
        real(wp), parameter :: tol = 1e-6
        if (any(abs(a - b) > tol)) then
            print *, "Assertion Failed: ", label
            print *, "Expected:", b
            print *, "Actual:", a
            stop 1
        else
            print *, "Assertion Passed: ", label
        end if
    end subroutine assert_almost_equal

    subroutine test_calc_stable_orbit()
        real(wp) :: r(1), theta(1)
        real(wp) :: pos(1, 2), vel(1, 2)
        real(wp) :: pos2(1, 2), vel2(1, 2)
        real(wp) :: pos3(1, 2), vel3(1, 2)

        r = 1.0_wp
        theta = 0.0_wp
        call calc_stable_orbit(r, theta, pos, vel)
        call assert_almost_equal(pos, reshape([0.0_wp, 1.0_wp], [1_wp, 2_wp]), "test_calc_stable_orbit pos 1")
        call assert_almost_equal(vel, reshape([-1.0_wp, 0.0_wp], [1_wp, 2_wp]), "test_calc_stable_orbit vel 1")

        theta = 2.0_wp * PI
        call calc_stable_orbit(r, theta, pos2, vel2)
        call assert_almost_equal(pos, pos2, "test_calc_stable_orbit pos 2")
        call assert_almost_equal(vel, vel2, "test_calc_stable_orbit vel 2")

        theta = PI
        call calc_stable_orbit(r, theta, pos3, vel3)
        call assert_almost_equal(pos, -1.0 * pos3, "test_calc_stable_orbit pos 3")
        call assert_almost_equal(vel, -1.0 * vel3, "test_calc_stable_orbit vel 3")
    end subroutine test_calc_stable_orbit

    subroutine test_calc_acc()
        real(wp) :: mass(2)
        real(wp) :: pos(2, 2)
        real(wp) :: acc(2, 2)
        real(wp) :: epsilon
        real(wp) :: expected_acc(2, 2)

        mass = [2.0, 0.5]
        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [1.0, 0.0]
        
        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc(acc, pos, mass)
        !$omp end target data
        epsilon = 1.1 * (2.0**(-0.48))

        expected_acc(1, :) = [1.0, 0.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[1.0, 0.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc horizontal")

        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [0.0, 1.0]

        call calc_acc(acc, pos, mass)
        expected_acc(1, :) = [0.0, 1.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[0.0, 1.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc vertical")
    end subroutine test_calc_acc

    subroutine test_advance_pos()
        real(wp) :: pos(1, 2)
        real(wp) :: pos_prev(1, 2)
        real(wp) :: pos_temp(1, 2)
        real(wp) :: acc(1, 2)
        real(wp) :: dt
        real(wp) :: expected_pos(1, 2)

        dt = 0.5
        pos(1, :) = [1.0, 2.0]
        pos_prev(1, :) = [0.5, 3.0]
        acc(1, :) = [0.5, -1.0]

        call advance_pos(acc, pos, pos_prev, pos_temp, dt)

        expected_pos(1, 1) = 2.0 - 0.5 + 0.5 * 0.5**2
        expected_pos(1, 2) = 4.0 - 3.0 + (-1.0) * 0.5**2

        call assert_almost_equal(pos, expected_pos, "test_advance_pos")
    end subroutine test_advance_pos

end module nbody_simulation

#ifdef MAIN
program main

    use nbody_simulation
    implicit none

    integer :: i
    integer :: n_particles
    integer :: steps_arg = 10, seed_arg, argc, iarg, stat
    logical :: seed_given
    integer :: seed_size
    integer, allocatable :: seed_array(:)
    character(len=64) :: arg

    steps_arg = 10
    seed_given = .false.

    argc = command_argument_count()
    iarg = 1
    do while (iarg <= argc)
        call get_command_argument(iarg, arg)
        select case (trim(arg))
        case ('-n', '--particles')
            iarg = iarg + 1
            call get_command_argument(iarg, arg, status=stat)
            if (stat /= 0) then
                print *, "Missing value for ", trim(arg)
                stop 1
            end if
            read(arg, *) n_particles
        case ('--steps')
            iarg = iarg + 1
            call get_command_argument(iarg, arg, status=stat)
            if (stat /= 0) then
                print *, "Missing value for --steps"
                stop 1
            end if
            read(arg, *) steps_arg
        case ('--seed')
            iarg = iarg + 1
            call get_command_argument(iarg, arg, status=stat)
            if (stat /= 0) then
                print *, "Missing value for --seed"
                stop 1
            end if
            read(arg, *) seed_arg
            seed_given = .true.
        case default
            print *, "Unknown argument: ", trim(arg)
            stop 1
        end select
        iarg = iarg + 1
    end do

    open(newunit=file_unit, file='final.csv', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
        print *, "Error opening final.csv"
    end if

    if (seed_given) then
        call random_seed(size=seed_size)
        allocate(seed_array(seed_size))
        seed_array = seed_arg
        call random_seed(put=seed_array)
        deallocate(seed_array)
    else
      call random_seed()
    end if


    call run_sim(n_particles, steps_arg)

    if (ios == 0) close(file_unit)
end program main
#endif

#ifdef TEST
program test

    use nbody_simulation
    implicit none

    call test_calc_stable_orbit()
    call test_calc_acc()
    call test_advance_pos()

end program test
#endif
