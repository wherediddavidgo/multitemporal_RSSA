import whitebox
import glob
import os
import argparse

def main(fdr_in_dir: str, fac_in_dir: str, fdr_out_dir: str, fac_out_dir: str):
    wbt = whitebox.WhiteboxTools()

    fdr_mos_out = os.path.join(fdr_out_dir, 'nhd_fdr_mos_test.tif')
    if not os.path.exists(fdr_mos_out):
        fdr_inputs = glob.glob(os.path.join(fdr_in_dir, 'fdr_*.tif'))
        fdr_inputs_str = fdr_inputs[0]
        for p in fdr_inputs[1:]:
            fdr_inputs_str = f'{fdr_inputs_str};{p}'

        wbt.mosaic(output=fdr_mos_out, inputs=fdr_inputs_str)
    else:
        print(f'{os.path.join(fdr_out_dir, 'nhd_fdr_mos.tif')} exists', flush=True)

    fac_mos_out = os.path.join(fac_out_dir, 'nhd_fac_mos_test.tif')
    if not os.path.exists(fac_mos_out):
        fac_inputs = glob.glob(os.path.join(fac_in_dir, 'fac_*.tif'))
        fac_inputs_str = fac_inputs[0]
        for p in fac_inputs[1:]:
            fac_inputs_str = f'{fac_inputs_str};{p}'

        wbt.mosaic(output=fac_mos_out, inputs=fac_inputs_str)
    else:
        print(f'{os.path.join(fac_out_dir, 'nhd_fac_mos.tif')} exists', flush=True)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--fdr-in-dir', required=True)
    ap.add_argument('--fac-in-dir', required=True)
    ap.add_argument('--fdr-out-dir', required=True)
    ap.add_argument('--fac-out-dir', required=True)
    main(**vars(ap.parse_args()))


