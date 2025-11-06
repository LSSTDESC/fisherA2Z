from fisherA2Z.fisher import Fisher

import pyccl as ccl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    cosmo = ccl.Cosmology(Omega_c=0.2666, 
                           Omega_b=0.049, 
                           h=0.6727, 
                           sigma8=0.831, 
                           n_s=0.9645, 
                           transfer_function='eisenstein_hu')
    
    probe = '3x2pt'
    
    base = Fisher(cosmo, outliers=[0.]*5,
                   zbias=[0.0]*5,
                   zvariance=[0.05, 0.05, 0.05, 0.05, 0.05], probe = probe, 
                   save_deriv=f'fisher_data/obj_deriv_{probe}_y10.pkl', overwrite=True)
    
    base.process()
    
    probe = 'ss'
    
    base = Fisher(cosmo, outliers=[0.]*5,
                   zbias=[0.0]*5,
                   zvariance=[0.05, 0.05, 0.05, 0.05, 0.05], probe = probe, 
                   save_deriv=f'fisher_data/obj_deriv_{probe}_y10.pkl', overwrite=True)
    
    base.process()
    
    probe = '3x2pt'
    
    base = Fisher(cosmo, outliers=[0.]*5,
                   zbias=[0.0]*5,
                   zvariance=[0.05, 0.05, 0.05, 0.05, 0.05], probe = probe, y1 = True,
                   save_deriv=f'fisher_data/obj_deriv_{probe}_y1.pkl', overwrite=True)
    
    base.process()
    
    probe = 'ss'
    
    base = Fisher(cosmo, outliers=[0.]*5,
                   zbias=[0.0]*5,
                   zvariance=[0.05, 0.05, 0.05, 0.05, 0.05], probe = probe, y1 = True, 
                   save_deriv=f'fisher_data/obj_deriv_{probe}_y1.pkl', overwrite=True)
    
    base.process()


from typing import Optional

def run(cfg_path: Optional[str] = None):
    """
    Wrapper for CLI: precompute Fisher matrices.
    If your implementation has a function named main() or main2(), call it here.
    """
    # Example: if your script defines main() with no args:
    try:
        main()                    # noqa: F821  (your script's main)
        return
    except NameError:
        pass

    # If you actually need cfg_path:
    try:
        main(cfg_path)            # noqa: F821
        return
    except NameError:
        raise RuntimeError("prepare_fisher.py: no callable main/main(cfg) found")

