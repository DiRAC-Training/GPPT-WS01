module packing_common
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

    interface
        subroutine c_exit(status) bind(C, name="exit")
            import :: c_int
            integer(c_int), value :: status
        end subroutine c_exit
    end interface

contains

    ! Check the error status of an MPI call and print a message on failure
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

end module packing_common

module packing_kernels
    use mpi_f08
    use packing_common
    implicit none

contains

    ! Fill the send buffer with a unique value per (rank, message, element) so
    ! that correctness can be verified after communication.
    subroutine init_kernel(data, rank, num_messages, msg_size, stride)
        real(rk), intent(inout) :: data(:, :)
        integer, intent(in) :: rank, num_messages, msg_size, stride
        integer :: m, i
        !$omp target teams distribute parallel do collapse(2)
        do m = 1, num_messages
            do i = 1, stride
                if (i <= msg_size) then
                    data(i, m) = real(rank * 1000000 + (m - 1) * 10000 + (i - 1), rk)
                else
                    data(i, m) = -1.0_rk
                end if
            end do
        end do
    end subroutine init_kernel

    ! TODO: Implement the pack kernel.
    ! Each thread copies one element from the strided source buffer (stride, num_messages)
    ! to the contiguous destination buffer (msg_size, num_messages):
    !   dst(i, m) = src(i, m)   for i = 1..msg_size, m = 1..num_messages
    ! Use a !$omp target teams distribute parallel do collapse(2) construct.
    subroutine pack_kernel(dst, src, num_messages, msg_size, stride)
        real(rk), intent(inout) :: dst(:, :)
        real(rk), intent(in) :: src(:, :)
        integer, intent(in) :: num_messages, msg_size, stride
        ! TODO
    end subroutine pack_kernel

    ! TODO: Implement the unpack kernel.
    ! Each thread copies one element from the contiguous source buffer (msg_size, num_messages)
    ! to the strided destination buffer (stride, num_messages):
    !   dst(i, m) = src(i, m)   for i = 1..msg_size, m = 1..num_messages
    ! Use a !$omp target teams distribute parallel do collapse(2) construct.
    subroutine unpack_kernel(dst, src, num_messages, msg_size, stride)
        real(rk), intent(inout) :: dst(:, :)
        real(rk), intent(in) :: src(:, :)
        integer, intent(in) :: num_messages, msg_size, stride
        ! TODO
    end subroutine unpack_kernel

    ! Check that the first msg_size elements of each strided message match the
    ! expected values from source_rank.
    function verify(h_data, source_rank, num_messages, msg_size, stride) result(correct)
        real(rk), intent(in) :: h_data(:, :)
        integer, intent(in) :: source_rank, num_messages, msg_size, stride
        integer :: correct
        integer :: m, i
        correct = 1
        do m = 1, num_messages
            do i = 1, msg_size
                if (h_data(i, m) /= real(source_rank * 1000000 + (m - 1) * 10000 + (i - 1), rk)) then
                    correct = 0
                    return
                end if
            end do
        end do
    end function verify

end module packing_kernels

program packing
    use mpi_f08
    use omp_lib
    use packing_common
    use packing_kernels
    implicit none

    integer :: size, rank, ierr
    integer :: num_messages, msg_size, stride, num_iters

    ! GPU binding via local rank
    integer :: local_rank
    type(MPI_Comm) :: local_comm
    integer :: num_devices

    real(rk), allocatable :: send(:, :), recv(:, :)
    real(rk), allocatable :: send_buf(:, :), recv_buf(:, :)

    integer :: next, prev, buf_size, packed_size
    integer :: iter, m
    real(trk) :: t0, t1, t2, t3
    integer :: correct_blocking, correct_packing, global_correct
    type(MPI_Status) :: status
    type(MPI_Request) :: reqs(2)
    type(MPI_Status) :: statuses(2)

    call MPI_Init(ierr)
    call mpi_check(ierr, "MPI_Init")
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call mpi_check(ierr, "MPI_Comm_rank")
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    call mpi_check(ierr, "MPI_Comm_size")

    num_messages = get_argval_int("-num_messages", 128)
    msg_size     = get_argval_int("-msg_size", 1024)
    stride       = get_argval_int("-stride", msg_size + 128)
    num_iters    = get_argval_int("-num_iters", 100)

    local_rank = -1
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, &
                             MPI_INFO_NULL, local_comm, ierr)
    call mpi_check(ierr, "MPI_Comm_split_type")
    call MPI_Comm_rank(local_comm, local_rank, ierr)
    call mpi_check(ierr, "MPI_Comm_rank")
    call MPI_Comm_free(local_comm, ierr)
    call mpi_check(ierr, "MPI_Comm_free")

    num_devices = omp_get_num_devices()
    if (num_devices > 0) then
        call omp_set_default_device(mod(local_rank, num_devices))
    end if

    ! Ring topology: send right, receive left
    next = mod(rank + 1, size)
    prev = mod(rank - 1 + size, size)

    buf_size    = num_messages * stride
    packed_size = num_messages * msg_size

    allocate (send(stride, num_messages))
    allocate (recv(stride, num_messages))
    allocate (send_buf(msg_size, num_messages))
    allocate (recv_buf(msg_size, num_messages))

    send = 0.0_rk
    recv = 0.0_rk
    send_buf = 0.0_rk
    recv_buf = 0.0_rk
    !$omp target enter data map(to: send, recv, send_buf, recv_buf)

    ! ================================================================
    ! Approach 1: Blocking MPI_Sendrecv — one call per message (baseline)
    ! ================================================================
    call init_kernel(send, rank, num_messages, msg_size, stride)
    recv = 0.0_rk
    !$omp target update to(recv)
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Barrier")

    t0 = MPI_Wtime()
    do iter = 1, num_iters
        !$omp target data use_device_addr(send, recv)
        do m = 1, num_messages
            call MPI_Sendrecv(send(1, m), msg_size, MPI_REAL_TYPE, next, 0, &
                              recv(1, m), msg_size, MPI_REAL_TYPE, prev, 0, &
                              MPI_COMM_WORLD, status, ierr)
            call mpi_check(ierr, "MPI_Sendrecv")
        end do
        !$omp end target data
    end do
    t1 = MPI_Wtime()

    !$omp target update from(recv(1:msg_size, 1:num_messages))
    correct_blocking = verify(recv, prev, num_messages, msg_size, stride)
    call MPI_Allreduce(correct_blocking, global_correct, 1, MPI_INTEGER, MPI_MIN, &
                       MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Allreduce")
    correct_blocking = global_correct

    ! ================================================================
    ! Approach 2: Packing kernel + non-blocking MPI (TODO)
    ! ================================================================
    call init_kernel(send, rank, num_messages, msg_size, stride)
    recv = 0.0_rk
    !$omp target update to(recv)
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Barrier")

    t2 = MPI_Wtime()
    do iter = 1, num_iters
        ! TODO: Call pack_kernel to gather send into send_buf
        !   call pack_kernel(send_buf, send, num_messages, msg_size, stride)

        ! TODO: Post MPI_Isend (send_buf -> next) and MPI_Irecv (recv_buf <- prev)
        !   inside a !$omp target data use_device_addr(send_buf, recv_buf) region.
        !   The signatures are
        !     MPI_Isend(buf, count, datatype, dest, tag, comm, request, ierror)
        !     MPI_Irecv(buf, count, datatype, source, tag, comm, request, ierror)
        !   Then call MPI_Waitall to complete both requests:
        !     MPI_Waitall(count, array_of_requests, array_of_statuses, ierror)

        ! TODO: Call unpack_kernel to scatter recv_buf into recv
        !   call unpack_kernel(recv, recv_buf, num_messages, msg_size, stride)
    end do
    t3 = MPI_Wtime()

    !$omp target update from(recv(1:msg_size, 1:num_messages))
    correct_packing = verify(recv, prev, num_messages, msg_size, stride)
    call MPI_Allreduce(correct_packing, global_correct, 1, MPI_INTEGER, MPI_MIN, &
                       MPI_COMM_WORLD, ierr)
    call mpi_check(ierr, "MPI_Allreduce")
    correct_packing = global_correct

    ! ================================================================
    ! Results
    ! ================================================================
    if (rank == 0) then
        write (*, '("Packing kernels: ",I0," messages x ",I0," elements (stride ",I0,"), ")') &
            num_messages, msg_size, stride
        write (*, '(I0," iterations, ",I0," GPUs")') num_iters, size
        if (correct_blocking == 1) then
            write (*, '("Blocking:  ",F8.4," s  (correct: yes)")') t1 - t0
        else
            write (*, '("Blocking:  ",F8.4," s  (correct: no)")') t1 - t0
        end if
        if (correct_packing == 1) then
            write (*, '("Packing:   ",F8.4," s  (correct: yes)")') t3 - t2
        else
            write (*, '("Packing:   ",F8.4," s  (correct: no)")') t3 - t2
        end if
        if (correct_packing == 1 .and. (t3 - t2) > 0) then
            write (*, '("Speedup:   ",F8.2)') (t1 - t0) / (t3 - t2)
        end if
    end if

    !$omp target exit data map(delete: send, recv, send_buf, recv_buf)
    deallocate (send, recv, send_buf, recv_buf)

    call MPI_Finalize(ierr)
    call mpi_check(ierr, "MPI_Finalize")
    call exit_with_status(0)
end program packing
