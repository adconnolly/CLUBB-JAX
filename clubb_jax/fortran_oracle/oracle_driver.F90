!===========================================================================
! oracle_driver.F90
!
! Standalone Fortran oracle: reads test parameters from stdin, calls the
! actual CLUBB diffusion_zt_lhs / diffusion_zm_lhs / zm2zt / zt2zm
! subroutines, and writes results to stdout as space-separated doubles.
!
! Protocol (stdin, one value per line):
!   nzm        (integer)
!   ngrdcol    (integer)
!   deltaz     (real)
!   zm_init    (real)
!   K_zm_val   (real, uniform K_zm for diffusion_zt_lhs)
!   K_zt_val   (real, uniform K_zt for diffusion_zm_lhs)
!   nu_val     (real, uniform nu)
!   rho_zm_val (real, uniform rho_ds_zm)
!   rho_zt_val (real, uniform rho_ds_zt)
!
! Output (stdout):
!   TAG values...
! where TAG is one of: ZM2ZT ZT2ZM DIFF_ZT_LHS DIFF_ZM_LHS
! followed by all elements in row-major order (ngrdcol * nzt or similar).
!===========================================================================
program oracle_driver

  use clubb_precision,    only: core_rknd
  use grid_class,         only: grid, zm2zt_api, zt2zm_api, ddzm, ddzt
  use diffusion,          only: diffusion_zt_lhs, diffusion_zm_lhs
  use tridiag_lu_solvers, only: tridiag_lu_solve

  implicit none

  ! --- Index constants matching diffusion.F90 ---
  integer, parameter :: kp1_tdiag = 1, k_tdiag = 2, km1_tdiag = 3
  integer, parameter :: kp1_mdiag = 1, k_mdiag = 2, km1_mdiag = 3

  ! Interpolation weight indices from grid_class.F90
  integer, parameter :: t_above = 1, t_below = 2
  integer, parameter :: m_above = 1, m_below = 2

  integer :: nzm, nzt, ngrdcol
  real(core_rknd) :: deltaz, zm_init
  real(core_rknd) :: K_zm_val, K_zt_val, nu_val, rho_zm_val, rho_zt_val

  real(core_rknd), allocatable :: zm(:,:), zt(:,:)
  real(core_rknd), allocatable :: invrs_dzt(:,:), invrs_dzm(:,:)
  real(core_rknd), allocatable :: K_zm(:,:), K_zt(:,:), nu(:)
  real(core_rknd), allocatable :: invrs_rho_ds_zt(:,:), rho_ds_zm(:,:)
  real(core_rknd), allocatable :: invrs_rho_ds_zm(:,:), rho_ds_zt(:,:)
  real(core_rknd), allocatable :: azm_test(:,:), azt_test(:,:)
  real(core_rknd), allocatable :: lhs_zt(:,:,:), lhs_zm(:,:,:)
  real(core_rknd), allocatable :: lhs_zt_copy(:,:,:), lhs_zm_copy(:,:,:)
  real(core_rknd), allocatable :: azt_out(:,:), azm_out(:,:)
  real(core_rknd), allocatable :: rhs_test_zt(:,:), soln_zt(:,:)
  real(core_rknd), allocatable :: rhs_test_zm(:,:), soln_zm(:,:)

  type(grid) :: gr
  integer :: i, k

  ! --- Read parameters ---
  read(*,*) nzm
  read(*,*) ngrdcol
  read(*,*) deltaz
  read(*,*) zm_init
  read(*,*) K_zm_val
  read(*,*) K_zt_val
  read(*,*) nu_val
  read(*,*) rho_zm_val
  read(*,*) rho_zt_val

  nzt = nzm - 1

  ! --- Allocate ---
  allocate( zm(ngrdcol, nzm), zt(ngrdcol, nzt) )
  allocate( invrs_dzt(ngrdcol, nzt), invrs_dzm(ngrdcol, nzm) )
  allocate( K_zm(ngrdcol, nzm), K_zt(ngrdcol, nzt), nu(ngrdcol) )
  allocate( invrs_rho_ds_zt(ngrdcol, nzt), rho_ds_zm(ngrdcol, nzm) )
  allocate( invrs_rho_ds_zm(ngrdcol, nzm), rho_ds_zt(ngrdcol, nzt) )
  allocate( azm_test(ngrdcol, nzm), azt_test(ngrdcol, nzt) )
  allocate( lhs_zt(3, ngrdcol, nzt), lhs_zm(3, ngrdcol, nzm) )
  allocate( lhs_zt_copy(3, ngrdcol, nzt), lhs_zm_copy(3, ngrdcol, nzm) )
  allocate( azt_out(ngrdcol, nzt), azm_out(ngrdcol, nzm) )
  allocate( rhs_test_zt(ngrdcol, nzt), soln_zt(ngrdcol, nzt) )
  allocate( rhs_test_zm(ngrdcol, nzm), soln_zm(ngrdcol, nzm) )

  ! --- Build uniform ascending grid ---
  ! zm(k) = zm_init + (k-1)*deltaz  (1-based Fortran k=1..nzm)
  do k = 1, nzm
    do i = 1, ngrdcol
      zm(i, k) = zm_init + real(k-1, core_rknd) * deltaz
    end do
  end do
  ! zt(k) = 0.5*(zm(k) + zm(k+1))  for k=1..nzt
  do k = 1, nzt
    do i = 1, ngrdcol
      zt(i, k) = 0.5_core_rknd * (zm(i,k) + zm(i,k+1))
    end do
  end do
  ! invrs_dzt(k) = 1/(zm(k+1)-zm(k))  for k=1..nzt
  do k = 1, nzt
    do i = 1, ngrdcol
      invrs_dzt(i,k) = 1.0_core_rknd / (zm(i,k+1) - zm(i,k))
    end do
  end do
  ! invrs_dzm interior k=2..nzm-1: 1/(zt(k)-zt(k-1))
  ! boundaries (k=1 and k=nzm) copy adjacent interior
  do k = 2, nzm-1
    do i = 1, ngrdcol
      invrs_dzm(i,k) = 1.0_core_rknd / (zt(i,k) - zt(i,k-1))
    end do
  end do
  do i = 1, ngrdcol
    invrs_dzm(i,1)   = invrs_dzm(i,2)
    invrs_dzm(i,nzm) = invrs_dzm(i,nzm-1)
  end do

  ! --- Constant coefficient fields ---
  K_zm(:,:) = K_zm_val
  K_zt(:,:) = K_zt_val
  nu(:)     = nu_val
  rho_ds_zm(:,:) = rho_zm_val
  rho_ds_zt(:,:) = rho_zt_val
  invrs_rho_ds_zt(:,:) = 1.0_core_rknd / rho_zm_val  ! rho on zt ≈ rho_zm in this test
  invrs_rho_ds_zm(:,:) = 1.0_core_rknd / rho_zt_val

  ! --- Build grid derived type ---
  gr%nzm = nzm
  gr%nzt = nzt
  gr%grid_dir_indx = 1
  gr%grid_dir      = 1.0_core_rknd
  gr%k_lb_zm = 1;  gr%k_ub_zm = nzm
  gr%k_lb_zt = 1;  gr%k_ub_zt = nzt

  allocate( gr%zm(ngrdcol, nzm), gr%zt(ngrdcol, nzt) )
  allocate( gr%invrs_dzt(ngrdcol, nzt), gr%invrs_dzm(ngrdcol, nzm) )
  allocate( gr%weights_zm2zt(ngrdcol, nzt, m_above:m_below) )
  allocate( gr%weights_zt2zm(ngrdcol, nzm, t_above:t_below) )

  gr%zm = zm;  gr%zt = zt
  gr%invrs_dzt = invrs_dzt;  gr%invrs_dzm = invrs_dzm

  ! weights_zm2zt: k=1..nzt, weight for azm(k+1) / azm(k)
  do k = 1, nzt
    do i = 1, ngrdcol
      gr%weights_zm2zt(i,k,m_above) = (zt(i,k) - zm(i,k)) / (zm(i,k+1) - zm(i,k))
      gr%weights_zm2zt(i,k,m_below) = 1.0_core_rknd - gr%weights_zm2zt(i,k,m_above)
    end do
  end do

  ! weights_zt2zm: k=1 lower boundary (copy), k=2..nzm-1 interior, k=nzm upper
  do i = 1, ngrdcol
    gr%weights_zt2zm(i,1,t_above) = 1.0_core_rknd
    gr%weights_zt2zm(i,1,t_below) = 0.0_core_rknd
  end do
  do k = 2, nzm-1
    do i = 1, ngrdcol
      gr%weights_zt2zm(i,k,t_above) = (zm(i,k) - zt(i,k-1)) / (zt(i,k) - zt(i,k-1))
      gr%weights_zt2zm(i,k,t_below) = 1.0_core_rknd - gr%weights_zt2zm(i,k,t_above)
    end do
  end do
  do i = 1, ngrdcol
    gr%weights_zt2zm(i,nzm,t_above) = (zm(i,nzm) - zt(i,nzt-1)) / (zt(i,nzt) - zt(i,nzt-1))
    gr%weights_zt2zm(i,nzm,t_below) = 1.0_core_rknd - gr%weights_zt2zm(i,nzm,t_above)
  end do

  ! --- Test fields for interpolation: linear profile f = 2 + 0.5*z ---
  do k = 1, nzm
    do i = 1, ngrdcol
      azm_test(i,k) = 2.0_core_rknd + 0.5_core_rknd * zm(i,k)
    end do
  end do
  do k = 1, nzt
    do i = 1, ngrdcol
      azt_test(i,k) = 2.0_core_rknd + 0.5_core_rknd * zt(i,k)
    end do
  end do

  ! --- Call zm2zt (zm2zt_api -> linear_interpolated_azt_2D for l_cubic_interp=F) ---
  azt_out = zm2zt_api(nzm, nzt, ngrdcol, gr, azm_test)
  write(*,'(A)',advance='no') 'ZM2ZT'
  do k = 1, nzt
    do i = 1, ngrdcol
      write(*,'(1X,E25.17)',advance='no') azt_out(i,k)
    end do
  end do
  write(*,*)

  ! --- Call zt2zm (zt2zm_api -> linear_interpolated_azm_2D for l_cubic_interp=F) ---
  azm_out = zt2zm_api(nzm, nzt, ngrdcol, gr, azt_test)
  write(*,'(A)',advance='no') 'ZT2ZM'
  do k = 1, nzm
    do i = 1, ngrdcol
      write(*,'(1X,E25.17)',advance='no') azm_out(i,k)
    end do
  end do
  write(*,*)

  ! --- Call diffusion_zt_lhs ---
  call diffusion_zt_lhs(nzm, nzt, ngrdcol, gr, K_zm, K_zt, nu, invrs_rho_ds_zt, rho_ds_zm, lhs_zt)
  write(*,'(A)',advance='no') 'DIFF_ZT_LHS'
  do k = 1, nzt
    do i = 1, ngrdcol
      write(*,'(3(1X,E25.17))',advance='no') lhs_zt(1,i,k), lhs_zt(2,i,k), lhs_zt(3,i,k)
    end do
  end do
  write(*,*)

  ! --- Call diffusion_zm_lhs ---
  call diffusion_zm_lhs(nzm, nzt, ngrdcol, gr, K_zt, K_zm, nu, invrs_rho_ds_zm, rho_ds_zt, lhs_zm)
  write(*,'(A)',advance='no') 'DIFF_ZM_LHS'
  do k = 1, nzm
    do i = 1, ngrdcol
      write(*,'(3(1X,E25.17))',advance='no') lhs_zm(1,i,k), lhs_zm(2,i,k), lhs_zm(3,i,k)
    end do
  end do
  write(*,*)

  ! --- Test tridiagonal solver on well-conditioned artificial system (nzt levels) ---
  ! System: main=4, super=-1 (except top), sub=-1 (except bottom), rhs=k
  ! This is a diagonally-dominant tridiagonal with a unique solution.
  do k = 1, nzt
    do i = 1, ngrdcol
      lhs_zt_copy(kp1_tdiag, i, k) = -1.0_core_rknd   ! superdiag
      lhs_zt_copy(k_tdiag,   i, k) =  4.0_core_rknd   ! main diag
      lhs_zt_copy(km1_tdiag, i, k) = -1.0_core_rknd   ! subdiag
      rhs_test_zt(i, k) = real(k, core_rknd)
    end do
  end do
  ! Zero boundary entries: super at top, sub at bottom
  do i = 1, ngrdcol
    lhs_zt_copy(kp1_tdiag, i, nzt) = 0.0_core_rknd
    lhs_zt_copy(km1_tdiag, i, 1)   = 0.0_core_rknd
  end do
  call tridiag_lu_solve(nzt, ngrdcol, lhs_zt_copy, rhs_test_zt, soln_zt)
  write(*,'(A)',advance='no') 'SOLVER_ZT_SOLN'
  do k = 1, nzt
    do i = 1, ngrdcol
      write(*,'(1X,E25.17)',advance='no') soln_zt(i,k)
    end do
  end do
  write(*,*)

  ! --- Test tridiagonal solver on well-conditioned artificial system (nzm levels) ---
  do k = 1, nzm
    do i = 1, ngrdcol
      lhs_zm_copy(kp1_mdiag, i, k) = -1.0_core_rknd
      lhs_zm_copy(k_mdiag,   i, k) =  4.0_core_rknd
      lhs_zm_copy(km1_mdiag, i, k) = -1.0_core_rknd
      rhs_test_zm(i, k) = real(k, core_rknd)
    end do
  end do
  do i = 1, ngrdcol
    lhs_zm_copy(kp1_mdiag, i, nzm) = 0.0_core_rknd
    lhs_zm_copy(km1_mdiag, i, 1)   = 0.0_core_rknd
  end do
  call tridiag_lu_solve(nzm, ngrdcol, lhs_zm_copy, rhs_test_zm, soln_zm)
  write(*,'(A)',advance='no') 'SOLVER_ZM_SOLN'
  do k = 1, nzm
    do i = 1, ngrdcol
      write(*,'(1X,E25.17)',advance='no') soln_zm(i,k)
    end do
  end do
  write(*,*)

  ! --- Cleanup ---
  deallocate(gr%zm, gr%zt, gr%invrs_dzt, gr%invrs_dzm)
  deallocate(gr%weights_zm2zt, gr%weights_zt2zm)

end program oracle_driver
