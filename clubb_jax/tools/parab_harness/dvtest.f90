program dvtest
  use Parabolic_constants, only: epss
  use parabolic, only: gamma, parab
  implicit none
  integer, parameter :: dp = selected_real_kind(p=12)
  real(dp), parameter :: pi_dp = 3.14159265358979323846_dp
  real(dp), parameter :: limit = 10.0_dp**308
  real(dp) :: order, argument, dv_lo, dv_hi
  integer :: ios
  do
    read(*,*,iostat=ios) order, argument
    if (ios /= 0) exit
    epss = 1.0e-4_dp
    dv_lo = dvf(order, argument)
    epss = 1.0e-15_dp
    dv_hi = dvf(order, argument)
    write(*,'(4ES28.18E3)') order, argument, dv_lo, dv_hi
  end do
contains
  function dvf(ord, arg) result(res)
    real(dp), intent(in) :: ord, arg
    real(dp) :: res
    real(dp), dimension(2) :: uaxx, vaxx
    integer :: ierr
    if (arg <= 0.0_dp) then
      call parab(-ord-0.5_dp, -arg, 0, uaxx, vaxx, ierr)
      res = vaxx(1) / ((1.0_dp/pi_dp)*gamma(-ord)) - sin(pi_dp*(-ord-0.5_dp))*uaxx(1)
    else
      call parab(-ord-0.5_dp, arg, 0, uaxx, vaxx, ierr)
      res = uaxx(1)
    end if
    if (ierr /= 0) res = limit
  end function dvf
end program dvtest
