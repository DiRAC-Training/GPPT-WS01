module solver_mod
  use precision_mod
  implicit none

contains

  ! Flat row-major index into an n-by-n matrix
  pure function idx(i, j, n) result(k)
    ! TODO (Step 1)
    integer, intent(in) :: i, j, n
    integer :: k
    k = (i - 1) * n + j
  end function idx

  ! Dense matrix-vector product: y = A * x.
  subroutine matvec(y, A, x, n)
    integer, intent(in) :: n
    real(wp), intent(out) :: y(n)
    real(wp), intent(in) :: A(n*n)
    real(wp), intent(in) :: x(n)
    integer :: i, j
    real(wp) :: s

    ! TODO (Step 1)
    !$omp parallel do private(j, s)
    do i = 1, n
      s = 0.0_wp
      do j = 1, n
        s = s + A(idx(i, j, n)) * x(j)
      end do
      y(i) = s
    end do
  end subroutine matvec

  ! Dot product: result = sum(a[i] * b[i]).
  function dot(a, b, n) result(sum_val)
    integer, intent(in) :: n
    real(wp), intent(in) :: a(n)
    real(wp), intent(in) :: b(n)
    real(wp) :: sum_val
    integer :: i

    sum_val = 0.0_wp
    ! TODO (Step 2)
    !$omp parallel do reduction(+:sum_val)
    do i = 1, n
      sum_val = sum_val + a(i) * b(i)
    end do
  end function dot

  ! AXPBY operation: y = alpha * x + beta * y.
  subroutine axpby(y, x, alpha, beta, n)
    integer, intent(in) :: n
    real(wp), intent(inout) :: y(n)
    real(wp), intent(in) :: x(n)
    real(wp), intent(in) :: alpha
    real(wp), intent(in) :: beta
    integer :: i

    ! TODO (Step 3): offload this loop to the GPU. Replace the directive
    ! below with an equivalent OpenMP target directive.
    !$omp parallel do
    do i = 1, n
      y(i) = alpha * x(i) + beta * y(i)
    end do
  end subroutine axpby

  ! Solve A*x = b using the conjugate gradient method.
  function cg_solve(x, A, b, n, max_iter) result(n_iter)
    integer, intent(in) :: n, max_iter
    real(wp), intent(inout) :: x(n)
    real(wp), intent(in) :: A(n*n)
    real(wp), intent(in) :: b(n)
    integer :: n_iter

    real(wp), allocatable :: r(:), p(:), A_times_p(:)
    real(wp) :: residual_sq_old, residual_sq_new
    real(wp) :: alpha, beta
    integer :: i
    real(wp), parameter :: EPS = epsilon(1.0_wp)

    allocate(r(n))
    allocate(p(n))
    allocate(A_times_p(n))

    ! TODO (Step 5)

    ! Step 1: r_0 = f - K*x_0
    call matvec(r, A, x, n)
    ! TODO (Step 4)
    !$omp parallel do
    do i = 1, n
      r(i) = b(i) - r(i)
    end do

    ! Step 2: p_0 = r_0
    ! TODO (Step 4)
    p = r

    residual_sq_old = dot(r, r, n)

    do n_iter = 0, max_iter - 1
      ! Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
      call matvec(A_times_p, A, p, n)
      alpha = residual_sq_old / dot(p, A_times_p, n)

      ! Step 3b: x_{k+1} = x_k + alpha_k * p_k
      call axpby(x, p, alpha, 1.0_wp, n)

      ! Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
      call axpby(r, A_times_p, -alpha, 1.0_wp, n)

      ! Step 3d: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
      !          p_{k+1} = r_{k+1} + beta_k * p_k
      residual_sq_new = dot(r, r, n)

      if (residual_sq_new < EPS) exit

      beta = residual_sq_new / residual_sq_old
      call axpby(p, r, 1.0_wp, beta, n)
      residual_sq_old = residual_sq_new

      print '(I0, A, E15.6)', n_iter, ': r = ', residual_sq_new / n
    end do
    ! TODO (Step 5)

    deallocate(r)
    deallocate(p)
    deallocate(A_times_p)
  end function cg_solve

end module solver_mod
