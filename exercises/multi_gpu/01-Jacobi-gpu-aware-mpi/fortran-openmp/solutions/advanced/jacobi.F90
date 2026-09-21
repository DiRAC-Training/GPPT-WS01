! Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! Modified work Copyright (c) 2026 Miren Radia
! Fortran + OpenMP target offload port of the CUDA C++ version.

module jacobi_common
    use mpi_f08
    use iso_c_binding, only: c_int
    use iso_fortran_env, only: error_unit, output_unit
    implicit none

#ifdef USE_DOUBLE
    integer, parameter :: rk = kind(1.0d0)
    type(MPI_Datatype), parameter :: MPI_REAL_TYPE = MPI_DOUBLE_PRECISION
#else
    integer, parameter :: rk = kind(1.0)
    type(MPI_Datatype), parameter :: MPI_REAL_TYPE = MPI_REAL
#endif
    ! Kind used for timings (MPI_Wtime returns double precision)
    integer, parameter :: trk = kind(1.0d0)

    real(rk), parameter :: tol = 1.0e-8_rk
    real(rk), parameter :: PI = 2.0_rk*asin(1.0_rk)

    interface
        subroutine c_exit(status) bind(C, name="exit")
            import :: c_int
            integer(c_int), value :: status
        end subroutine c_exit
    end interface

contains

    ! Check the error status of an MPI call and print a message on failure
    ! (Fortran replacement for the MPI_CALL macro in the C++ version)
    subroutine mpi_check(ierr, call_name)
        integer, intent(in) :: ierr
        character(len=*), intent(in) :: call_name
        character(len=MPI_MAX_ERROR_STRING) :: err_string
        integer :: err_len, ierr2
        if (ierr /= MPI_SUCCESS) then
            call MPI_Error_string(ierr, err_string, err_len, ierr2)
            write (error_unit, '(A,A,A,A)') "ERROR: MPI call ", call_name, &
                " failed with ", err_string(1:err_len)
        end if
    end subroutine mpi_check

    ! Cleanly exit with a given exit status (flushes Fortran I/O first)
    subroutine exit_with_status(status)
        integer, intent(in) :: status
        flush (output_unit)
        flush (error_unit)
        call c_exit(int(status, c_int))
    end subroutine exit_with_status

    ! Get the integer value following a named command line argument
    ! (replacement for the get_argval template in the C++ version)
    function get_argval_int(arg, default_val) result(argval)
        character(len=*), intent(in) :: arg
        integer, intent(in) :: default_val
        integer :: argval
        integer :: i, ios
        character(len=64) :: carg
        argval = default_val
        do i = 1, command_argument_count() - 1
            call get_command_argument(i, carg)
            if (trim(carg) == arg) then
                call get_command_argument(i + 1, carg)
                read (carg, *, iostat=ios) argval
                if (ios /= 0) argval = default_val
                return
            end if
        end do
    end function get_argval_int

    ! Check if a flag is present on the command line
    ! (replacement for get_arg in the C++ version)
    function get_arg(arg) result(found)
        character(len=*), intent(in) :: arg
        logical :: found
        integer :: i
        character(len=64) :: carg
        found = .false.
        do i = 1, command_argument_count()
            call get_command_argument(i, carg)
            if (trim(carg) == arg) then
                found = .true.
                return
            end if
        end do
    end function get_arg

end module jacobi_common

module jacobi_kernels
    use mpi_f08
    use jacobi_common
    implicit none

contains

    ! Set dirichlet boundary conditions on the left and right border.
    ! A row a(:, iy) is contiguous in memory, so it can be sent/received
    ! directly with MPI.
    subroutine initialize_boundaries(a, a_new, pi, offset, nx, my_ny, ny)
        real(rk), intent(inout) :: a(:, :), a_new(:, :)
        real(rk), intent(in) :: pi
        integer, intent(in) :: offset, nx, my_ny, ny
        integer :: iy
        real(rk) :: y0

        !$omp target teams distribute parallel do
        do iy = 1, my_ny
            y0 = sin(2.0_rk*pi*(offset + iy)/(ny - 1))
            a(1, iy) = y0
            a(nx, iy) = y0
            a_new(1, iy) = y0
            a_new(nx, iy) = y0
        end do
    end subroutine initialize_boundaries

    ! One Jacobi iteration. The l2 norm of the residue is only computed if
    ! calculate_norm is true (like the CUDA kernel in the C++ version).
    subroutine jacobi_kernel(a_new, a, l2_norm, iy_start, iy_end, nx, calculate_norm)
        real(rk), intent(inout) :: a_new(:, :)
        real(rk), intent(in) :: a(:, :)
        real(rk), intent(out) :: l2_norm
        integer, intent(in) :: iy_start, iy_end, nx
        logical, intent(in) :: calculate_norm
        integer :: ix, iy
        real(rk) :: new_val, residue

        l2_norm = 0.0_rk
        if (calculate_norm) then
            !$omp target teams distribute parallel do collapse(2) reduction(+:l2_norm)
            do iy = iy_start, iy_end - 1
                do ix = 2, nx - 1
                    new_val = 0.25_rk*(a(ix + 1, iy) + a(ix - 1, iy) + &
                                       a(ix, iy + 1) + a(ix, iy - 1))
                    a_new(ix, iy) = new_val
                    residue = new_val - a(ix, iy)
                    l2_norm = l2_norm + residue*residue
                end do
            end do
        else
            !$omp target teams distribute parallel do collapse(2)
            do iy = iy_start, iy_end - 1
                do ix = 2, nx - 1
                    a_new(ix, iy) = 0.25_rk*(a(ix + 1, iy) + a(ix - 1, iy) + &
                                             a(ix, iy + 1) + a(ix, iy - 1))
                end do
            end do
        end if
    end subroutine jacobi_kernel

    ! Single GPU Jacobi solver used to compute the reference solution and
    ! runtime (runs on every rank, on the GPU assigned to that rank)
    function single_gpu(nx, ny, iter_max, a_ref_h, nccheck, print_flag) result(runtime)
        integer, intent(in) :: nx, ny, iter_max, nccheck
        real(rk), intent(out) :: a_ref_h(:, :)
        logical, intent(in) :: print_flag
        real(trk) :: runtime

        real(rk), allocatable :: a(:, :), a_new(:, :), tmp(:, :)
        real(rk) :: l2_norm, l2_norm_local
        logical :: calculate_norm
        integer :: iy_start, iy_end, ix, iter
        real(trk) :: start, stop

        iy_start = 2
        iy_end = ny

        allocate (a(nx, ny), a_new(nx, ny))
        a = 0.0_rk
        a_new = 0.0_rk

        !$omp target enter data map(to: a, a_new)

        ! Set dirichlet boundary conditions on left and right border
        call initialize_boundaries(a, a_new, PI, 0, nx, ny, ny)

        if (print_flag) then
            write (*, '(A,I0,A,I0,A,I0,A,I0,A)') "Single GPU jacobi relaxation: ", &
                iter_max, " iterations on ", ny, " x ", nx, &
                " mesh with norm check every ", nccheck, " iterations"
        end if

        iter = 0
        l2_norm = 1.0_rk

        start = MPI_Wtime()
        do while (l2_norm > tol .and. iter < iter_max)
            calculate_norm = (mod(iter, nccheck) == 0) .or. (mod(iter, 100) == 0)
            call jacobi_kernel(a_new, a, l2_norm_local, iy_start, iy_end, nx, calculate_norm)

            ! Apply periodic boundary conditions
            !$omp target teams distribute parallel do
            do ix = 1, nx
                a_new(ix, 1) = a_new(ix, iy_end - 1)
            end do
            !$omp target teams distribute parallel do
            do ix = 1, nx
                a_new(ix, iy_end) = a_new(ix, iy_start)
            end do

            if (calculate_norm) then
                l2_norm = sqrt(l2_norm_local)
                if (print_flag .and. mod(iter, 100) == 0) then
                    write (*, '(I5,", ",F0.6)') iter, l2_norm
                end if
            end if

            ! Swap a and a_new. The OpenMP device mapping is keyed on the host
            ! base address of the allocation which move_alloc preserves, so the
            ! swapped arrays resolve to the existing device allocations.
            call move_alloc(a, tmp)
            call move_alloc(a_new, a)
            call move_alloc(tmp, a_new)
            iter = iter + 1
        end do
        stop = MPI_Wtime()

        !$omp target update from(a)
        a_ref_h = a
        !$omp target exit data map(delete: a, a_new)
        deallocate (a, a_new)

        runtime = stop - start
    end function single_gpu

end module jacobi_kernels

program jacobi
    use mpi_f08
    use omp_lib
    use jacobi_common
    use jacobi_kernels
    implicit none

    integer :: size, rank, ierr

    integer :: iter_max, nccheck, nx, ny
    logical :: csv

    ! This code (below) gets your local rank on a node
    integer :: local_rank
    type(MPI_Comm) :: local_comm

    integer :: num_devices

    real(rk), allocatable :: a_ref_h(:, :), a_h(:, :)
    real(trk) :: runtime_serial

    integer :: chunk_size
    integer :: chunk_size_low, chunk_size_high, num_ranks_low
    real(rk), allocatable :: a(:, :), a_new(:, :), tmp(:, :)

    integer :: iy_start_global, iy_end_global, iy_start, iy_end
    integer :: iter, ix, iy, n_rows
    integer :: top, bottom
    type(MPI_Status) :: status
    real(rk) :: l2_norm, l2_norm_global
    logical :: calculate_norm  ! whether l2 norm will be calculated in an iteration or not
    real(trk) :: start, stop
    integer :: result_correct, global_result_correct

    call MPI_Init(ierr)
    call mpi_check(ierr, "MPI_Init")
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call mpi_check(ierr, "MPI_Comm_rank")
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    call mpi_check(ierr, "MPI_Comm_size")

    iter_max = get_argval_int("-niter", 1000)
    nccheck = get_argval_int("-nccheck", 1)
    nx = get_argval_int("-nx", 16384)
    ny = get_argval_int("-ny", 16384)
    csv = get_arg("-csv")

    local_rank = -1
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, &
                             MPI_INFO_NULL, local_comm, ierr)
    call mpi_check(ierr, "MPI_Comm_split_type")
    call MPI_Comm_rank(local_comm, local_rank, ierr)
    call mpi_check(ierr, "MPI_Comm_rank")
    call MPI_Comm_free(local_comm, ierr)
    call mpi_check(ierr, "MPI_Comm_free")

    num_devices = 0
    ! TODO: Get the available GPU devices into `num_devices` and use it and the
    ! local rank to set the active GPU with OpenMP routines (see omp_lib).
    num_devices = omp_get_num_devices()
    if (num_devices > 0) then
        call omp_set_default_device(mod(local_rank, num_devices))
    end if

    allocate (a_ref_h(nx, ny))
    allocate (a_h(nx, ny))
    runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, &
                                .not. csv .and. (0 == rank))

    chunk_size_low = (ny - 2)/size
    chunk_size_high = chunk_size_low + 1
    ! To calculate the number of ranks that need to compute an extra row,
    ! the following formula is derived from this equation:
    ! num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    num_ranks_low = size*chunk_size_low + size - &
                    (ny - 2)  ! Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low) then
        chunk_size = chunk_size_low
    else
        chunk_size = chunk_size_high
    end if

    allocate (a(nx, chunk_size + 2), a_new(nx, chunk_size + 2))
    a = 0.0_rk
    a_new = 0.0_rk

    ! My start index in the global array
    if (rank < num_ranks_low) then
        iy_start_global = rank*chunk_size_low + 2
    else
        iy_start_global = num_ranks_low*chunk_size_low + &
                          (rank - num_ranks_low)*chunk_size_high + 2
    end if
    iy_end_global = iy_start_global + chunk_size - 1  ! My last index in the global array
    iy_start = 2                                      ! My local start index for computation
    iy_end = iy_start + chunk_size                    ! My local last index

    !$omp target enter data map(to: a, a_new)

    ! Set dirichlet boundary conditions on left and right border
    call initialize_boundaries(a, a_new, PI, iy_start_global - 2, nx, chunk_size + 2, ny)

    if (.not. csv .and. 0 == rank) then
        write (*, '(A,I0,A,I0,A,I0,A,I0,A)') "Jacobi relaxation: ", iter_max, &
            " iterations on ", ny, " x ", nx, &
            " mesh with norm check every ", nccheck, " iterations"
    end if

    iter = 0
    l2_norm = 1.0_rk

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Barrier")
    start = MPI_Wtime()
    do while (l2_norm > tol .and. iter < iter_max)
        calculate_norm = (mod(iter, nccheck) == 0) .or. (.not. csv .and. (mod(iter, 100) == 0))

        call jacobi_kernel(a_new, a, l2_norm, iy_start, iy_end, nx, calculate_norm)

        ! TODO: Compute top and bottom neighbor, use reflecting/periodic boundaries.
        ! This means rank 0 and rank (size-1) exchange data
        top = merge(rank - 1, size - 1, rank > 0)
        bottom = mod(rank + 1, size)

        ! TODO: Use MPI_Sendrecv to exchange the data with the top and bottom neighbors
        ! Use GPU-aware MPI here, i.e. receive the data directly in a_new on the GPU
        ! and send it from there without manually copying the data. Wrap the MPI
        ! calls in an `!$omp target data use_device_addr(a_new)` region so that
        ! device pointers are passed to MPI.

        ! The first newly calculated row (`iy_start`) is sent to the top neighbour
        ! and the bottom boundary row (`iy_end`) is received from the bottom process.
        ! The last calculated row (`iy_end-1`) is sent to the bottom process and the
        ! top boundary (`1`) is received from the top.
        ! The `!$omp target teams distribute parallel do` constructs in jacobi_kernel
        ! are synchronous, so the computation on the GPU has completed before the
        ! data transfer starts.
        !$omp target data use_device_addr(a_new)
        call MPI_Sendrecv(a_new(1, iy_start), nx, MPI_REAL_TYPE, top, 0, &
                          a_new(1, iy_end), nx, MPI_REAL_TYPE, bottom, 0, &
                          MPI_COMM_WORLD, status, ierr)
        call mpi_check(ierr, "MPI_Sendrecv")
        call MPI_Sendrecv(a_new(1, iy_end - 1), nx, MPI_REAL_TYPE, bottom, 0, &
                          a_new(1, 1), nx, MPI_REAL_TYPE, top, 0, &
                          MPI_COMM_WORLD, status, ierr)
        call mpi_check(ierr, "MPI_Sendrecv")
        !$omp end target data

        if (calculate_norm) then
            call MPI_Allreduce(l2_norm, l2_norm_global, 1, MPI_REAL_TYPE, MPI_SUM, &
                               MPI_COMM_WORLD, ierr)
            call mpi_check(ierr, "MPI_Allreduce")
            l2_norm = sqrt(l2_norm_global)

            if (.not. csv .and. 0 == rank .and. mod(iter, 100) == 0) then
                write (*, '(I5,", ",F0.6)') iter, l2_norm
            end if
        end if

        ! Swap a and a_new. The OpenMP device mapping is keyed on the host base
        ! address of the allocation which move_alloc preserves, so the swapped
        ! arrays resolve to the existing device allocations.
        call move_alloc(a, tmp)
        call move_alloc(a_new, a)
        call move_alloc(tmp, a_new)
        iter = iter + 1
    end do
    stop = MPI_Wtime()

    ! Copy the result back to the host for the correctness check
    n_rows = min(ny - iy_start_global, chunk_size)
    !$omp target update from(a(1:nx, 2:n_rows + 1))
    a_h(:, iy_start_global:iy_start_global + n_rows - 1) = a(:, 2:n_rows + 1)
    !$omp target exit data map(delete: a, a_new)
    deallocate (a, a_new)

    result_correct = 1
    row_loop: do iy = iy_start_global, iy_end_global - 1
        do ix = 2, nx - 1
            if (abs(a_ref_h(ix, iy) - a_h(ix, iy)) > tol) then
                write (error_unit, '(A,I0,A,I0,A,I0,A,I0,A,F0.6,A,F0.6,A)') &
                    "ERROR on rank ", rank, ": a[", iy, " * ", nx, " + ", ix, "] = ", &
                    a_h(ix, iy), " does not match ", a_ref_h(ix, iy), " (reference)"
                result_correct = 0
                exit row_loop
            end if
        end do
    end do row_loop

    call MPI_Allreduce(result_correct, global_result_correct, 1, MPI_INTEGER, MPI_MIN, &
                       MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Allreduce")
    result_correct = global_result_correct

    if (rank == 0 .and. result_correct == 1) then
        if (csv) then
            write (*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,F0.6,A,F0.6)') "mpi, ", nx, ", ", ny, ", ", &
                iter_max, ", ", nccheck, ", ", size, ", 1, ", stop - start, ", ", runtime_serial
        else
            write (*, '(A,I0,A)') "Num GPUs: ", size, "."
            write (*, '(I0,A,I0,A,F8.4,A,I0,A,F8.4,A,F8.2,A,F8.2,A)') &
                ny, "x", nx, ": 1 GPU: ", runtime_serial, " s, ", size, " GPUs: ", &
                stop - start, " s, speedup: ", runtime_serial/(stop - start), &
                ", efficiency: ", runtime_serial/(size*(stop - start))*100, " %"
        end if
    end if

    deallocate (a_ref_h, a_h)

    call MPI_Finalize(ierr)
    call mpi_check(ierr, "MPI_Finalize")
    call exit_with_status(merge(0, 1, result_correct == 1))
end program jacobi
