! iso_c_binding interfaces to the cuBLAS v2 C API. NVIDIA-only; the amdflang
! path would bind the rocBLAS analogs with the same shape.
module cublas_mod
  use iso_c_binding, only: c_int, c_float, c_ptr
  implicit none

  integer(c_int), parameter :: CUBLAS_OP_N = 0
  integer(c_int), parameter :: CUBLAS_OP_T = 1
  integer(c_int), parameter :: CUBLAS_FILL_MODE_LOWER = 0
  integer(c_int), parameter :: CUBLAS_FILL_MODE_UPPER = 1
  integer(c_int), parameter :: CUBLAS_STATUS_SUCCESS = 0

  interface
    integer(c_int) function cublasCreate(handle) bind(C, name="cublasCreate_v2")
      import :: c_int, c_ptr
      type(c_ptr), intent(out) :: handle
    end function

    integer(c_int) function cublasDestroy(handle) bind(C, name="cublasDestroy_v2")
      import :: c_int, c_ptr
      type(c_ptr), value :: handle
    end function

    ! y = alpha*op(A)*x + beta*y. cuBLAS reads A column-major; our A is row-major
    ! (= A^T to cuBLAS), so op=CUBLAS_OP_T gives A*x. Default host pointer mode:
    ! alpha/beta are host scalars, A/x/y are device pointers.
    integer(c_int) function cublasSgemv(handle, trans, m, n, alpha, A, lda, &
        x, incx, beta, y, incy) bind(C, name="cublasSgemv_v2")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: handle
      integer(c_int), value :: trans, m, n, lda, incx, incy
      real(c_float) :: alpha, beta
      real(c_float) :: A(*), x(*), y(*)
    end function

    ! Symmetric matvec y = alpha*A*x + beta*y, reads only one triangle. A is
    ! symmetric so the row-major/column-major transpose is a no-op here; with A
    ! fully stored either fill mode is correct. Halves matrix traffic vs Sgemv.
    integer(c_int) function cublasSsymv(handle, uplo, n, alpha, A, lda, &
        x, incx, beta, y, incy) bind(C, name="cublasSsymv_v2")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: handle
      integer(c_int), value :: uplo, n, lda, incx, incy
      real(c_float) :: alpha, beta
      real(c_float) :: A(*), x(*), y(*)
    end function

    ! res written to host memory; call blocks until it is ready.
    integer(c_int) function cublasSdot(handle, n, x, incx, y, incy, res) &
        bind(C, name="cublasSdot_v2")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: handle
      integer(c_int), value :: n, incx, incy
      real(c_float) :: x(*), y(*)
      real(c_float) :: res
    end function

    integer(c_int) function cublasSaxpy(handle, n, alpha, x, incx, y, incy) &
        bind(C, name="cublasSaxpy_v2")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: handle
      integer(c_int), value :: n, incx, incy
      real(c_float) :: alpha
      real(c_float) :: x(*), y(*)
    end function

    integer(c_int) function cublasSscal(handle, n, alpha, x, incx) &
        bind(C, name="cublasSscal_v2")
      import :: c_int, c_float, c_ptr
      type(c_ptr), value :: handle
      integer(c_int), value :: n, incx
      real(c_float) :: alpha
      real(c_float) :: x(*)
    end function
  end interface

end module cublas_mod
