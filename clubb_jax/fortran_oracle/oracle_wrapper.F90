!===========================================================================
! oracle_wrapper.F90
!
! Thin Fortran wrapper exposing CLUBB grid/diffusion routines via plain arrays.
! Designed for F2PY: no derived types in any public interface.
!
! Wraps (non-upwind path only):
!   - diffusion_zt_lhs  (diffusion.F90)
!   - diffusion_zm_lhs  (diffusion.F90)
!   - linear_interpolated_azt_2D (grid_class.F90)  [zm2zt]
!   - linear_interpolated_azm_2D (grid_class.F90)  [zt2zm]
!
! The grid derived type is constructed internally from plain arrays.
!===========================================================================
module oracle_wrapper

  use clubb_precision, only: core_rknd
  use grid_class,      only: grid
  use diffusion,       only: diffusion_zt_lhs, diffusion_zm_lhs

  implicit none

  private
  public :: oracle_diffusion_zt_lhs, oracle_diffusion_zm_lhs, &
            oracle_zm2zt, oracle_zt2zm

contains

  !---------------------------------------------------------------------------
  ! Build a minimal grid derived type from flat arrays.
  ! Only populates the fields accessed by the non-upwind diffusion path:
  !   gr%invrs_dzt, gr%invrs_dzm
  ! and the fields accessed by the interpolation routines:
  !   gr%zm, gr%zt, gr%weights_zm2zt, gr%weights_zt2zm, gr%grid_dir_indx
  !---------------------------------------------------------------------------
  subroutine make_grid(nzm, nzt, ngrdcol, zm_in, zt_in, &
                       invrs_dzt_in, invrs_dzm_in, gr)
    integer,         intent(in)  :: nzm, nzt, ngrdcol
    real(core_rknd), intent(in)  :: zm_in(ngrdcol, nzm)
    real(core_rknd), intent(in)  :: zt_in(ngrdcol, nzt)
    real(core_rknd), intent(in)  :: invrs_dzt_in(ngrdcol, nzt)
    real(core_rknd), intent(in)  :: invrs_dzm_in(ngrdcol, nzm)
    type(grid),      intent(out) :: gr
    integer :: i, k

    gr%nzm = nzm
    gr%nzt = nzt
    gr%grid_dir_indx = 1     ! ascending grid
    gr%grid_dir      = 1.0_core_rknd
    gr%k_lb_zm = 1;  gr%k_ub_zm = nzm
    gr%k_lb_zt = 1;  gr%k_ub_zt = nzt

    allocate( gr%zm(ngrdcol, nzm), gr%zt(ngrdcol, nzt) )
    allocate( gr%invrs_dzt(ngrdcol, nzt), gr%invrs_dzm(ngrdcol, nzm) )
    allocate( gr%weights_zm2zt(ngrdcol, nzt, 1:2) )
    allocate( gr%weights_zt2zm(ngrdcol, nzm, 1:2) )

    gr%zm       = zm_in
    gr%zt       = zt_in
    gr%invrs_dzt = invrs_dzt_in
    gr%invrs_dzm = invrs_dzm_in

    ! Compute interpolation weights (ascending grid only)
    ! weights_zm2zt(i, k, m_above=1): weight for azm(k+1)  [upper zm level]
    ! weights_zm2zt(i, k, m_below=2): weight for azm(k)    [lower zm level]
    ! Fortran 1-based: k=1..nzt
    do k = 1, nzt
      do i = 1, ngrdcol
        gr%weights_zm2zt(i,k,1) = &
          (gr%zt(i,k) - gr%zm(i,k)) / (gr%zm(i,k+1) - gr%zm(i,k))
        gr%weights_zm2zt(i,k,2) = 1.0_core_rknd - gr%weights_zm2zt(i,k,1)
      end do
    end do

    ! weights_zt2zm(i, k, t_above=1): weight for azt(k)
    ! weights_zt2zm(i, k, t_below=2): weight for azt(k-1)
    ! k=1 (lower boundary): ascending => azm(1) = azt(1), so t_above=1, t_below=0
    do i = 1, ngrdcol
      gr%weights_zt2zm(i,1,1) = 1.0_core_rknd
      gr%weights_zt2zm(i,1,2) = 0.0_core_rknd
    end do
    ! k=2..nzm-1 interior
    do k = 2, nzm-1
      do i = 1, ngrdcol
        gr%weights_zt2zm(i,k,1) = &
          (gr%zm(i,k) - gr%zt(i,k-1)) / (gr%zt(i,k) - gr%zt(i,k-1))
        gr%weights_zt2zm(i,k,2) = 1.0_core_rknd - gr%weights_zt2zm(i,k,1)
      end do
    end do
    ! k=nzm upper boundary: linear extension
    do i = 1, ngrdcol
      gr%weights_zt2zm(i,nzm,1) = &
        (gr%zm(i,nzm) - gr%zt(i,nzt-1)) / (gr%zt(i,nzt) - gr%zt(i,nzt-1))
      gr%weights_zt2zm(i,nzm,2) = 1.0_core_rknd - gr%weights_zt2zm(i,nzm,1)
    end do

  end subroutine make_grid


  subroutine free_grid(gr)
    type(grid), intent(inout) :: gr
    if (allocated(gr%zm))           deallocate(gr%zm)
    if (allocated(gr%zt))           deallocate(gr%zt)
    if (allocated(gr%invrs_dzt))    deallocate(gr%invrs_dzt)
    if (allocated(gr%invrs_dzm))    deallocate(gr%invrs_dzm)
    if (allocated(gr%weights_zm2zt)) deallocate(gr%weights_zm2zt)
    if (allocated(gr%weights_zt2zm)) deallocate(gr%weights_zt2zm)
  end subroutine free_grid


  !---------------------------------------------------------------------------
  ! Public: wrap diffusion_zt_lhs
  !   lhs(3, ngrdcol, nzt)  -- [superdiag, maindiag, subdiag]
  !---------------------------------------------------------------------------
  subroutine oracle_diffusion_zt_lhs( nzm, nzt, ngrdcol, &
                                       zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, &
                                       K_zm, K_zt, nu, invrs_rho_ds_zt, rho_ds_zm, &
                                       lhs )
    integer,         intent(in)  :: nzm, nzt, ngrdcol
    real(core_rknd), intent(in)  :: zm_in(ngrdcol,nzm), zt_in(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: invrs_dzt_in(ngrdcol,nzt), invrs_dzm_in(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: K_zm(ngrdcol,nzm), K_zt(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: nu(ngrdcol)
    real(core_rknd), intent(in)  :: invrs_rho_ds_zt(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: rho_ds_zm(ngrdcol,nzm)
    real(core_rknd), intent(out) :: lhs(3,ngrdcol,nzt)

    type(grid) :: gr

    call make_grid(nzm, nzt, ngrdcol, zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, gr)
    call diffusion_zt_lhs(nzm, nzt, ngrdcol, gr, K_zm, K_zt, nu, invrs_rho_ds_zt, rho_ds_zm, lhs)
    call free_grid(gr)

  end subroutine oracle_diffusion_zt_lhs


  !---------------------------------------------------------------------------
  ! Public: wrap diffusion_zm_lhs
  !   lhs(3, ngrdcol, nzm)  -- [superdiag, maindiag, subdiag]
  !---------------------------------------------------------------------------
  subroutine oracle_diffusion_zm_lhs( nzm, nzt, ngrdcol, &
                                       zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, &
                                       K_zt, K_zm, nu, invrs_rho_ds_zm, rho_ds_zt, &
                                       lhs )
    integer,         intent(in)  :: nzm, nzt, ngrdcol
    real(core_rknd), intent(in)  :: zm_in(ngrdcol,nzm), zt_in(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: invrs_dzt_in(ngrdcol,nzt), invrs_dzm_in(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: K_zt(ngrdcol,nzt), K_zm(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: nu(ngrdcol)
    real(core_rknd), intent(in)  :: invrs_rho_ds_zm(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: rho_ds_zt(ngrdcol,nzt)
    real(core_rknd), intent(out) :: lhs(3,ngrdcol,nzm)

    type(grid) :: gr

    call make_grid(nzm, nzt, ngrdcol, zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, gr)
    call diffusion_zm_lhs(nzm, nzt, ngrdcol, gr, K_zt, K_zm, nu, invrs_rho_ds_zm, rho_ds_zt, lhs)
    call free_grid(gr)

  end subroutine oracle_diffusion_zm_lhs


  !---------------------------------------------------------------------------
  ! Public: wrap linear_interpolated_azt_2D  (zm -> zt, i.e. zm2zt)
  !---------------------------------------------------------------------------
  subroutine oracle_zm2zt( nzm, nzt, ngrdcol, &
                            zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, &
                            azm, azt )
    integer,         intent(in)  :: nzm, nzt, ngrdcol
    real(core_rknd), intent(in)  :: zm_in(ngrdcol,nzm), zt_in(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: invrs_dzt_in(ngrdcol,nzt), invrs_dzm_in(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: azm(ngrdcol,nzm)
    real(core_rknd), intent(out) :: azt(ngrdcol,nzt)

    type(grid) :: gr

    call make_grid(nzm, nzt, ngrdcol, zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, gr)
    call linear_interpolated_azt_2D(nzm, nzt, ngrdcol, gr, azm, azt)
    call free_grid(gr)

  end subroutine oracle_zm2zt


  !---------------------------------------------------------------------------
  ! Public: wrap linear_interpolated_azm_2D  (zt -> zm, i.e. zt2zm)
  !---------------------------------------------------------------------------
  subroutine oracle_zt2zm( nzm, nzt, ngrdcol, &
                            zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, &
                            azt, azm )
    integer,         intent(in)  :: nzm, nzt, ngrdcol
    real(core_rknd), intent(in)  :: zm_in(ngrdcol,nzm), zt_in(ngrdcol,nzt)
    real(core_rknd), intent(in)  :: invrs_dzt_in(ngrdcol,nzt), invrs_dzm_in(ngrdcol,nzm)
    real(core_rknd), intent(in)  :: azt(ngrdcol,nzt)
    real(core_rknd), intent(out) :: azm(ngrdcol,nzm)

    type(grid) :: gr

    call make_grid(nzm, nzt, ngrdcol, zm_in, zt_in, invrs_dzt_in, invrs_dzm_in, gr)
    call linear_interpolated_azm_2D(nzm, nzt, ngrdcol, gr, azt, azm)
    call free_grid(gr)

  end subroutine oracle_zt2zm

end module oracle_wrapper
