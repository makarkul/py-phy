import multiprocessing
import numpy as np
import time
import utils

def sl_mmse_equ_leaf(args):
    (hx, rx, nv_matrix, valid_res) = args
    ntx, nrx, nre = hx.shape

    eq_gain = np.full((ntx, nre), 0, dtype=np.csingle)
    eq_outp = np.full((ntx, nre), 0, dtype=np.csingle)

    for re in range(0, valid_res):
        h = hx[:, :, re]
        A = np.matmul(np.conjugate(h), np.transpose(h)) 
        tempM = np.add(A, nv_matrix)
        B = np.linalg.inv(tempM)
        eq_gain[:, re] = np.clip(
                            np.real(
                                np.diag(
                                    np.matmul(B, A)
                                    )
                                ), -1e3, 0.9999
                            )

        BH = np.matmul(B, np.conjugate(h))
        r = rx[:, re]
        eq_outp[:, re] = np.matmul(BH, r)

    return (eq_gain, eq_outp)

def sl_mmse_equ(rx_data, H, noise_var, params):
    nslots = params["nslots"]
    ntx = params["ntx"]
    nrx = params["nrx"]
    nre = params["nre"]
    nsym = params["nsym"]
    valid_res = params["valid_res"]
    data_syms = params["data_syms"]

    eq_gain_all = np.empty((0, 0), dtype=np.single)
    eq_outp_all = np.empty((0, 0), dtype=np.csingle)

    hx_slot_len = nsym * ntx * nrx * nre
    rx_slot_len = nsym * nrx * nre
    nv_slot_len = ntx * 2 # noise variance is stored as  [snr, noise_var]

    for slot in range(nslots):
        hx = H[slot * hx_slot_len : (slot + 1) * hx_slot_len].reshape(nsym, ntx, 
                nrx, nre)
        rx = rx_data[slot * rx_slot_len : (slot + 1) * rx_slot_len].reshape(nsym, 
                nrx, nre)
        nv = noise_var[slot * nv_slot_len : (slot + 1) * nv_slot_len].reshape(ntx, 2)

        nv_matrix = np.multiply(nv[0][1], np.eye(ntx, dtype=np.csingle))

        eq_gain_sym = np.full((nsym, ntx, nre), 0, dtype=np.csingle)
        eq_outp_sym  = np.full((nsym, ntx, nre), 0, dtype=np.csingle)

        for sym in range(nsym):
            if data_syms[sym]:
                _eq_gain = np.full((ntx, nre), 0, dtype=np.csingle)
                _eq_outp = np.full((ntx, nre), 0, dtype=np.csingle)
                step_size = valid_res >> 2
   
                with multiprocessing.Pool(processes=4) as pool:
                    results = pool.map(sl_mmse_equ_leaf, [(hx[sym, :, :, re:re+step_size], rx[sym, :, re:re+step_size],
                        nv_matrix, step_size) for re in range(0, valid_res, step_size)])

                    for i, (_tmp_gain, _tmp_outp) in enumerate(results):
                        _eq_gain[:, step_size*i:step_size*(i+1)] = _tmp_gain
                        _eq_outp[:, step_size*i:step_size*(i+1)] = _tmp_outp

                eq_gain_sym[sym, :, :] = _eq_gain
                eq_outp_sym[sym, :, :] = _eq_outp

        eq_gain_all = np.append(eq_gain_all, eq_gain_sym)
        eq_outp_all = np.append(eq_outp_all, eq_outp_sym)

    return eq_gain_all, eq_outp_all

def check_mmse_eq(val, ref):
    v_abserr = np.abs(ref - val)
    v_absref = np.abs(ref)
    max_val = v_abserr.max()
    max_loc = v_abserr.argmax()
    
    print(
        f"  Max absolute ratio - "
        f"val: {max_val:e} loc: {max_loc} ref: {v_absref[np.unravel_index(max_loc, v_absref.shape)]:e}"
        )
    snr_num = np.linalg.norm(ref)
    snr_den = np.linalg.norm(v_abserr)
    if snr_den:
        snr = 10 * np.log10(snr_num/snr_den)
        print(f"  SNR: {snr:f} dB")
    else:
        print("   Bit-exact")


if __name__ == '__main__':
    rx_data = utils.read_file('..\\test\\data\\Dmp_01.bin', np.csingle)
    H = utils.read_file('..\\test\data\\Dmp_02.bin', np.csingle)
    noise_var = utils.read_file('..\\test\\data\\Dmp_03.bin', np.float32)

    params = {"nslots": 4,  #change to 1, if only 1 dump is needed
              "ntx": 8,
              "nrx": 8,
              "nre": 3280,
              "nsym": 14,
              "valid_res": 3264, #272 * 12
              "data_syms": [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
              }

    start = time.perf_counter()
    gain, out = sl_mmse_equ(rx_data, H, noise_var, params)
    end = time.perf_counter()

    ref = np.transpose(
            np.reshape(
                utils.read_file('..\\test\\data\\Dmp_08.out', dtype = np.csingle), 
                (-1, params["nre"])
            )
          )
    check_mmse_eq(np.transpose(np.reshape(out, (-1, params["nre"]))), ref)

    print("Elapsed = {}s".format((end - start)))
