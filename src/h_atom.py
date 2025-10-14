# coding: utf-8
"""
@File        :   h_atom.py
@Time        :   2025/10/13 14:16:51
@Author      :   Usercyk
@Description :   Hartree Fock, VMC and DMC of H atom.
"""
import os
from functools import lru_cache

import pandas as pd
import pyqmc.api as pyq

from pyscf import gto, scf
from pyqmc.method.dmc import rundmc

HARTREE_TO_EV = 27.211386245988


@lru_cache
def run_mean_field():
    """
    Run mean field.
    """

    basename = os.path.splitext(os.path.basename(__file__))[0]

    # 拼接保存路径
    save_path = os.path.join("data", f"{basename}.mf.hdf5")

    mole = gto.M(
        atom="H 0. 0. 0.",
        basis="cc-pvtz",
        unit="Bohr",
        spin=1
    )

    mean_field = scf.UHF(mole)
    mean_field.chkfile = save_path
    mean_field.kernel()
    mean_field.dump_chk(mean_field.chkfile)
    return save_path


def slater_jastrow():
    """
    To slater jastrow
    """
    chkfile = run_mean_field()

    mol, mf = pyq.recover_pyscf(chkfile)  # type: ignore pylint: disable=W0632

    wf, to_opt = pyq.generate_wf(mol,
                                 mf,
                                 slater_kws={"optimize_orbitals": True,
                                             "optimize_zeros": False,
                                             "optimize_determinants": False},
                                 jastrow_kws={"na": 0})
    return mol, wf, to_opt


def run_vmc():
    """
    Run VMC
    """
    mol, wf, _ = slater_jastrow()

    nconf = 500
    nsteps = 300
    warmup = 30

    coords = pyq.initial_guess(mol, nconf)
    df, coords = pyq.vmc(
        wf,
        coords,
        nblocks=int(nsteps / 30),
        nsteps_per_block=30,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
    )

    df = pd.DataFrame(df)["energytotal"][int(warmup / 30):]
    e_mean_ha = df.mean()
    e_err_ha = df.sem()

    e_mean_ev = e_mean_ha * HARTREE_TO_EV
    e_err_ev = e_err_ha * HARTREE_TO_EV  # type: ignore
    print(
        f"VMC energy = {e_mean_ha:.6f} ± {e_err_ha:.6f} Ha = {e_mean_ev:.6f} ± {e_err_ev:.6f} eV")


def run_dmc():
    """
    Run DMC
    """
    mol, wf, _ = slater_jastrow()

    nconf = 1000
    nsteps = 300
    warmup = 30

    coords = pyq.initial_guess(mol, nconf)
    df, coords, _ = rundmc(
        wf,
        coords,
        nblocks=int(nsteps / 30),
        nsteps_per_block=30,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
        vmc_warmup=warmup
    )

    df = pd.DataFrame(df)["energytotal"][int(warmup / 30):]
    e_mean_ha = df.mean()
    e_err_ha = df.sem()

    e_mean_ev = e_mean_ha * HARTREE_TO_EV
    e_err_ev = e_err_ha * HARTREE_TO_EV  # type: ignore
    print(
        f"DMC energy = {e_mean_ha:.6f} ± {e_err_ha:.6f} Ha = {e_mean_ev:.6f} ± {e_err_ev:.6f} eV")


if __name__ == "__main__":
    run_vmc()
    run_dmc()
