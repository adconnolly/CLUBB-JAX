module clubb_precision
  implicit none
  public
  integer, parameter :: dp = selected_real_kind( p=12 )
  integer, parameter :: sp = selected_real_kind( p=6 )
  integer, parameter :: core_rknd = dp
end module clubb_precision
