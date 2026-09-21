module solver_mod
  use iso_c_binding
  use cublas_mod
  implicit none

  type(c_ptr) :: handle = c_null_ptr

contains

  pure function idx(i, j, n) result(k)
    !$omp declare target
    integer, intent(in) :: i, j, n
    integer :: k
    k = (i - 1) * n + j
  end function idx

  subroutine ensure_handle()
    if (.not. c_associated(handle)) then
      if (cublasCreate(handle) /= CUBLAS_STATUS_SUCCESS) error stop "cublasCreate failed"
    end if
  end subroutine ensure_handle

  ! y = A * x. A is row-major, so cuBLAS sees A^T => CUBLAS_OP_T recovers A*x.
  subroutine matvec(y, A, x, n)
    integer, intent(in) :: n
    real, intent(out) :: y(n)
    real, intent(in) :: A(n*n)
    real, intent(in) :: x(n)
    real(c_float) :: one, zero
    integer(c_int) :: stat

    one = 1.0; zero = 0.0
    call ensure_handle()
    !$omp target data use_device_ptr(A, x, y)
    stat = cublasSgemv(handle, CUBLAS_OP_T, n, n, one, A, n, x, 1, zero, y, 1)
    !$omp end target data
  end subroutine matvec

  function dot(a, b, n) result(res)
    integer, intent(in) :: n
    real, intent(in) :: a(n)
    real, intent(in) :: b(n)
    real :: res
    real(c_float) :: r
    integer(c_int) :: stat

    call ensure_handle()
    r = -1.0
    !$omp target data use_device_ptr(a, b)
    stat = cublasSdot(handle, n, a, 1, b, 1, r)
    !$omp end target data
    if (stat /= CUBLAS_STATUS_SUCCESS) print *, 'Sdot status =', stat
    res = r

    !!$omp target data use_device_ptr(a, b)
    !stat = cublasSdot(handle, n, a, 1, b, 1, r)
    !!$omp end target data
    !res = r
  end function dot

  ! y = alpha * x + beta * y.
  subroutine axpby(y, x, alpha, beta, n)
    integer, intent(in) :: n
    real, intent(inout) :: y(n)
    real, intent(in) :: x(n)
    real, intent(in) :: alpha
    real, intent(in) :: beta
    real(c_float) :: a, b
    integer(c_int) :: stat

    a = alpha; b = beta
    call ensure_handle()
    !$omp target data use_device_ptr(x, y)
    if (b /= 1.0) stat = cublasSscal(handle, n, b, y, 1)
    stat = cublasSaxpy(handle, n, a, x, 1, y, 1)
    !$omp end target data
  end subroutine axpby

  ! Solve A*x = b using the conjugate gradient method.
  function cg_solve(x, A, b, n, max_iter) result(n_iter)
    integer, intent(in) :: n, max_iter
    real, intent(inout) :: x(n)
    real, intent(in) :: A(n*n)
    real, intent(in) :: b(n)
    integer :: n_iter

    real, allocatable :: r(:), p(:), A_times_p(:)
    real :: residual_sq_old, residual_sq_new
    real :: alpha, beta
    integer :: i
    real, parameter :: EPS = epsilon(1.0)

    allocate(r(n))
    allocate(p(n))
    allocate(A_times_p(n))

    ! Map the solver's temporary arrays.
    !$omp target data map(alloc: r(1:n), p(1:n), A_times_p(1:n))

    ! Step 1: r_0 = f - K*x_0
    call matvec(r, A, x, n)
    !$omp target teams distribute parallel do
    do i = 1, n
      r(i) = b(i) - r(i)
    end do

    ! Step 2: p_0 = r_0
    !$omp target teams distribute parallel do
    do i = 1, n
      p(i) = r(i)
    end do

    residual_sq_old = dot(r, r, n)

    do n_iter = 0, max_iter - 1
      ! Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
      call matvec(A_times_p, A, p, n)
      alpha = residual_sq_old / dot(p, A_times_p, n)

      ! Step 3b: x_{k+1} = x_k + alpha_k * p_k
      call axpby(x, p, alpha, 1.0, n)

      ! Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
      call axpby(r, A_times_p, -alpha, 1.0, n)

      ! Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
      !          p_{k+1} = r_{k+1} + beta_k * p_k
      residual_sq_new = dot(r, r, n)
      if (residual_sq_new < EPS) exit

      beta = residual_sq_new / residual_sq_old
      call axpby(p, r, 1.0, beta, n)
      residual_sq_old = residual_sq_new

      print '(I0, A, E15.6)', n_iter, ': r = ', residual_sq_new / n
    end do
    !$omp end target data

    deallocate(r)
    deallocate(p)
    deallocate(A_times_p)
  end function cg_solve

end module solver_mod
