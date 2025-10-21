import os
import dirichlet
import numpy as np
import subprocess
import tempfile

eps = 1e-3

class GapPriors:
    def __init__(self, tmm: float, tmi: float, tmd: float,
                 tim: float, tii: float,
                 tdm: float, tdd: float):
        self.tmm     = tmm
        self.tmi     = tmi
        self.tmd     = tmd
        self.tim     = tim
        self.tii     = tii
        self.tdm     = tdm
        self.tdd     = tdd

def get_hmm_data(msa: str, priors: GapPriors, method: str) -> list[list[float]]:
    # write HMM to a temporary file using the priors passed in
    out_hmm = tempfile.NamedTemporaryFile(delete=False, suffix=".hmm")
    out_hmm.close()

    if priors:
        cmd = [
            "./src/hmmbuild",
            "--enone",
            "--set-tmm", str(priors.tmm),
            "--set-tmi", str(priors.tmi),
            "--set-tmd", str(priors.tmd),
            "--set-tim", str(priors.tim),
            "--set-tii", str(priors.tii),
            "--set-tdm", str(priors.tdm),
            "--set-tdd", str(priors.tdd),
            out_hmm.name,
            msa,
        ]
    else:
        cmd = [
            "./src/hmmbuild",
            "--pnone",
            "--enone",
            out_hmm.name,
            msa,
        ]

    # Add --return-counts only for Easel mode (no empty string when using dirichlet)
    if method == "easel":
        cmd.insert(1, "--return-counts")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"hmmbuild failed: {e}")
    
    # parse transition rows from the HMM file (each transition row has 7 columns)
    probs = []
    with open(out_hmm.name, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == "//":          # end of one HMM; keep going to read the next
                continue
            parts = s.split()
            if len(parts) != 7:
                continue
            try:
                vals = np.array([float(v) for v in parts], dtype=float)
            except ValueError:
                # skip non-numeric lines
                continue
            # file stores negative log-probabilities => convert to normal probs
            if method == "dirichlet":
                p = np.exp(-vals)
            else:  # method == "easel"
                p = vals
            probs.append(p.tolist())

    # clean up temporary HMM file
    try:
        os.unlink(out_hmm.name)
    except OSError:
        pass

    print(f"Read {len(probs)} transition probability rows from HMM.")

    return probs

def get_new_gap_priors(gp: GapPriors, probs: list[list[float]]) -> GapPriors:

    a_d_g = np.array([prob[0:3] for prob in probs])
    b_bp  = np.array([prob[3:5] for prob in probs])
    e_ep  = np.array([prob[5:7] for prob in probs])
    
    try:
        a = dirichlet.mle(a_d_g)
        b = dirichlet.mle(b_bp)
        e = dirichlet.mle(e_ep)
    except Exception as ex:
        print("-"*100)
        print(f"Exception during Dirichlet MLE: {ex}")
        print(f"Most likely we're simply done: Returning previous gap priors...")
        print("-"*100)
        return gp
    
    return GapPriors(
        tmm=a[0], tmi=a[1], tmd=a[2],
        tim=b[0], tii=b[1],
        tdm=e[0], tdd=e[1]
    )

def get_new_gap_priors_easel(gp: GapPriors, probs: list[list[float]]) -> GapPriors:

    # write probs to temporary CSV files
    tmp_m = tempfile.NamedTemporaryFile(delete=False, suffix="_m.csv")
    tmp_m.close()
    tmp_i = tempfile.NamedTemporaryFile(delete=False, suffix="_i.csv")
    tmp_i.close()
    tmp_d = tempfile.NamedTemporaryFile(delete=False, suffix="_d.csv")
    tmp_d.close()

    np.savetxt(tmp_m.name, np.array(probs)[:,:3], delimiter=" ", fmt="%.8f")
    np.savetxt(tmp_i.name, np.array(probs)[:,3:5], delimiter=" ", fmt="%.8f")
    np.savetxt(tmp_d.name, np.array(probs)[:,5:7], delimiter=" ", fmt="%.8f")

    tmp_out_m = tempfile.NamedTemporaryFile(delete=False, suffix="_m_out")
    tmp_out_m.close()
    tmp_out_i = tempfile.NamedTemporaryFile(delete=False, suffix="_i_out")
    tmp_out_i.close()
    tmp_out_d = tempfile.NamedTemporaryFile(delete=False, suffix="_d_out")
    tmp_out_d.close()

    # run Easel's dirichlet tool on each file
    cmd_m = ["./easel/miniapps/esl-mixdchlet", "fit", "1", "3", tmp_m.name, tmp_out_m.name]
    cmd_i = ["./easel/miniapps/esl-mixdchlet", "fit", "1", "2", tmp_i.name, tmp_out_i.name]
    cmd_d = ["./easel/miniapps/esl-mixdchlet", "fit", "1", "2", tmp_d.name, tmp_out_d.name]

    try:
        subprocess.run(cmd_m, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_i, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_d, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Easel dirichlet tool failed: {e}")
    print("Successfully fit Dirichlet mixtures using Easel.")

    def _parse_alphas(path: str, expected_dim: int) -> np.ndarray:
        with open(path, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"Unexpected output format in {path}: need 2 lines")
        parts = [float(x) for x in lines[1].split()]
        if len(parts) != expected_dim + 1:
            raise RuntimeError(
                f"Unexpected number of fields in {path}: got {len(parts)}, expected {expected_dim + 1}"
            )
        # drop the first field (mixture weight), keep the alphas
        return np.array(parts[1:], dtype=float)

    a = _parse_alphas(tmp_out_m.name, 3)  # tmm, tmi, tmd
    b = _parse_alphas(tmp_out_i.name, 2)  # tim, tii
    e = _parse_alphas(tmp_out_d.name, 2)  # tdm, tdd

    # cleanup temporary files
    for p in (
        tmp_m.name, tmp_i.name, tmp_d.name,
        tmp_out_m.name, tmp_out_i.name, tmp_out_d.name
    ):
        try:
            os.unlink(p)
        except OSError:
            pass

    return GapPriors(
        tmm=a[0], tmi=a[1], tmd=a[2],
        tim=b[0], tii=b[1],
        tdm=e[0], tdd=e[1]
    )
    
def get_max_diff(gp1: GapPriors, gp2: GapPriors) -> float:
    diffs = [
        abs(gp1.tmm     - gp2.tmm),
        abs(gp1.tmi     - gp2.tmi),
        abs(gp1.tmd     - gp2.tmd),
        abs(gp1.tim     - gp2.tim),
        abs(gp1.tii     - gp2.tii),
        abs(gp1.tdm     - gp2.tdm),
        abs(gp1.tdd     - gp2.tdd),
    ]
    return max(diffs)

def print_gap_priors(gp: GapPriors):
    print(f" tmm:     {gp.tmm:.6f}")
    print(f" tmi:     {gp.tmi:.6f}")
    print(f" tmd:     {gp.tmd:.6f}")
    print(f" tim:     {gp.tim:.6f}")
    print(f" tii:     {gp.tii:.6f}")
    print(f" tdm:     {gp.tdm:.6f}")
    print(f" tdd:     {gp.tdd:.6f}")

if __name__ == "__main__":
    msa = "Pfam-A.stk"
    print(f"Looking at file: {msa}")

    # set to true to run with no initial priors, for one iteration only
    no_priors = True

    # set this to "dirichlet" to use Dirichlet MLE
    # set this to "easel" to use Easel's built-in method
    method = "easel"

    # These are the initial gap priors for protein sequences,
    # originally trained by Graeme Mitchison on an early version of Pfam.
    gp = GapPriors(
        tmm=0.7939, tmi=0.0278, tmd=0.0135,
        tim=0.1331, tii=0.9002,
        tdm=0.5630, tdd=0.9002
    )
    
    max_iterations = 100
    if no_priors:
        max_iterations = 1
    
    for iteration in range(max_iterations):
        
        if no_priors:
            probs = get_hmm_data(msa, None, method)
        else:
            probs = get_hmm_data(msa, gp, method)
        
        # pseudocounts are not taken into account by hmmbuild when using --return-counts
        # we add them in here manually for Easel method
        if not no_priors and method == "easel":
            for row in probs:
                row[0] = row[0] + gp.tmm
                row[1] = row[1] + gp.tmi
                row[2] = row[2] + gp.tmd
                row[3] = row[3] + gp.tim
                row[4] = row[4] + gp.tii
                row[5] = row[5] + gp.tdm
                row[6] = row[6] + gp.tdd

        if method == "dirichlet":
            new_gp = get_new_gap_priors(gp, probs)
        elif method == "easel":
            new_gp = get_new_gap_priors_easel(gp, probs)
        else:
            raise ValueError(f"Unknown method: {method}")

        max_diff = get_max_diff(gp, new_gp)
        
        gp = new_gp
        
        print(f"Iteration {iteration+1}:")
        print_gap_priors(gp)
        
        if max_diff < eps:
            print("Converged.")
            break
    
    print()
    print("="*100)
    print()
    print(f"Final gap priors after {iteration+1} iterations:")
    print_gap_priors(gp)