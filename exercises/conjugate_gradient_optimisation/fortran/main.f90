program main
  use solver_mod
  use test_mod
  use precision_mod
  implicit none

  integer, parameter :: n = 8192
  integer, parameter :: cg_max_iter = 32
  logical, parameter :: RANDOMISE_SEED = .true.
  real(wp), parameter :: DIAG_SCALE = -1.0_wp 
  ! integer, parameter :: n = 32000
  ! integer, parameter :: cg_max_iter = 320
  ! logical, parameter :: RANDOMISE_SEED = .false.
  ! real,    parameter :: DIAG_SCALE = 0.42

  real(wp), allocatable :: A(:)
  real(wp), allocatable :: x_soln(:), b(:), x(:)
  integer :: iters
  real(wp) :: av_error2
  integer :: i

  ! For timing
  integer(8) :: count1, count2, count_rate
  integer(8) :: duration_us

  if (.not. run_tests()) then
    print *, "A test failed!"
    stop 1
  end if

  allocate(A(n * n)) ! note n*n
  allocate(x_soln(n))
  allocate(b(n))
  allocate(x(n))

  ! Seed the RNG 
  call init_rng(RANDOMISE_SEED)

  ! Initial conditions
  call generate_positive_definite(A, n)
  print *, "Generating random solution"
  call fill_rand_vec(x_soln, n, -1.0_wp, 1.0_wp)
  print *, "Generating right hand side"
  call calc_b(b, A, x_soln, n)
  
  x = 0.0_wp

  !$omp target data map(to: A(1:n*n), b(1:n)) &
  !$omp             map(tofrom: x(1:n))

  ! Solve
  call system_clock(count1, count_rate)
  iters = cg_solve(x, A, b, n, cg_max_iter)
  call system_clock(count2)

  !$omp end target data
  
  duration_us = ((count2 - count1) * 1000000_8) / count_rate

  print *, "Performed ", iters, " iterations"
  print *, "Solve time: ", duration_us, " us"
  if (iters > 0) print *, "Time per iteration: ", duration_us / iters, " us"

  av_error2 = 0.0_wp
  do i = 1, n
    av_error2 = av_error2 + abs(x_soln(i) - x(i))
  end do
  av_error2 = av_error2 / n

  print *, "Average error = ", sqrt(av_error2)

  deallocate(A)
  deallocate(x_soln)
  deallocate(b)
  deallocate(x)

contains

  function run_tests() result(all_passed)
    logical :: all_passed
    all_passed = .true.
    
    if (.not. test_matvec_identity()) then
      all_passed = .false.
      print *, "matvec identity failed!"
    end if

    if (.not. test_matvec_simple()) then
      all_passed = .false.
      print *, "matvec simple failed!"
    end if

    if (.not. test_dot()) then
      all_passed = .false.
      print *, "dot failed!"
    end if
  end function run_tests

  subroutine calc_b(b, A, x, n)
    integer, intent(in) :: n
    real(wp), intent(out) :: b(n)
    real(wp), intent(in) :: A(n*n), x(n)
    integer :: i, j

    !$omp parallel do private(j)
    do i = 1, n
      b(i) = 0.0_wp
      do j = 1, n
        b(i) = b(i) + A(idx(i, j, n)) * x(j)
      end do
    end do
  end subroutine calc_b

  ! Seed the RNG 
  subroutine init_rng(randomise)
    logical, intent(in) :: randomise
    integer :: seed_size, base, i
    integer, allocatable :: seed(:)

    call random_seed(size = seed_size)
    allocate(seed(seed_size))

    if (randomise) then
      call system_clock(count = base)
    else
      base = 42
    end if

    ! Spread the base value across the seed array
    seed = base + 37 * [(i, i = 0, seed_size - 1)]
    call random_seed(put = seed)

    deallocate(seed)
  end subroutine init_rng

  subroutine fill_rand_vec(x, size, min_val, max_val)
    integer, intent(in) :: size
    real(wp), intent(out) :: x(size)
    real(wp), intent(in) :: min_val, max_val
    integer :: i

    call random_number(x)
    !$omp parallel do
    do i = 1, size
      x(i) = min_val + x(i) * (max_val - min_val)
    end do
  end subroutine fill_rand_vec


  ! The matrix is stored row-major in a flat 1D array (see idx): row i occupies
  ! A((i-1)*n+1 : i*n), so a row is contiguous in memory. This may look
  ! surprising in Fortran, where a matrix is normally a 2D array -- but a Fortran
  ! 2D array is column-major, so matvec's row-wise product would stride through
  ! memory. We would otherwise have to store A transposed to make the rows
  ! contiguous; a flat row-major buffer gives that layout directly, so matvec's
  ! inner loop vectorises and its rows parallelise over threads.
  subroutine generate_positive_definite(A, n)
    integer, intent(in) :: n
    real(wp), intent(out) :: A(n*n)
    real(wp), allocatable :: B(:)
    real(wp) :: diag_shift
    integer :: i, j

    print *, "Generating matrix"
    allocate(B(n * n))
    call random_number(B)

    ! Make a symmetric matrix
    !$omp parallel do private(j)
    do i = 1, n
      do j = 1, n
        A(idx(i, j, n)) = (B(idx(i, j, n)) + B(idx(j, i, n))) / 2.0_wp
      end do
    end do

    ! Diagonal shift to make it positive definite. DIAG_SCALE > 0 uses
    ! f*sqrt(n) (barely SPD, ill-conditioned => more iterations); otherwise n/32.
    if (DIAG_SCALE > 0.0_wp) then
      diag_shift = DIAG_SCALE * sqrt(real(n, wp))
    else
      diag_shift = max(real(n, wp) / 32.0_wp, 1.0_wp)
    end if
    do i = 1, n
      A(idx(i, i, n)) = A(idx(i, i, n)) + diag_shift
    end do

    deallocate(B)
  end subroutine generate_positive_definite

end program main
