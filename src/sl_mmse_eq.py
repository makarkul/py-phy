import multiprocessing
import numpy as np
import time
import utils

def sl_mmse_eq_slot_leaf(args):
    hx, rx, nv_matrix, slot, valid_res, data_syms = args
    nsym, ntx, nrx, nre = hx.shape

    eq_gain_slot = np.full((nsym, ntx, nre), 0, dtype=np.csingle)
    eq_outp_slot = np.full((nsym, ntx, nre), 0, dtype=np.csingle)

    '''
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(sl_mmse_equ_sym_leaf, [(hx[sym, :, :, :],
                            rx[sym, :, :],
                            nv_matrix,
                            sym,
                            valid_res,
                            data_syms[sym]) for sym in range(nsym)]
                        )
        for i, result in enumerate(results):
            (eq_gain_slot[i, :, :], eq_outp_slot[i, :, :]) = result
        
    '''
    for sym in range(nsym):
        (eq_gain_slot[sym, :, :], eq_outp_slot[sym, :, :]) =\
            sl_mmse_equ_sym_leaf(
                    (hx[sym, :, :, :],
                     rx[sym, :, :],
                     nv_matrix,
                     sym,
                     valid_res,
                     data_syms[sym])
                    )

    return (eq_gain_slot, eq_outp_slot)


def sl_mmse_equ_sym_leaf(args):
    hx, rx, nv_matrix, sym, valid_res, data_sym = args
    ntx, nrx, nre = hx.shape

    eq_gain_sym = np.full((ntx, nre), 0, dtype=np.csingle)
    eq_outp_sym = np.full((ntx, nre), 0, dtype=np.csingle)

    if data_sym:
        step_size = valid_res >> 2

        for re in range(0, valid_res, step_size):
            eq_gain_sym[:, re:re+step_size], eq_outp_sym[:, re:re+step_size] =\
                sl_mmse_equ_re_leaf(
                    (hx[:, :, re:re+step_size],
                        rx[:, re:re+step_size],
                        nv_matrix,
                        step_size)
                    )

    return (eq_gain_sym, eq_outp_sym)

def sl_mmse_equ_re_leaf(args):
    hx, rx, nv_matrix, valid_res = args
    ntx, nrx, nre = hx.shape

    eq_gain_valid_res = np.full((ntx, valid_res), 0, dtype=np.csingle)
    eq_outp_valid_res = np.full((ntx, valid_res), 0, dtype=np.csingle)

    for re in range(0, valid_res):
        h = hx[:, :, re]
        A = np.matmul(np.conjugate(h), np.transpose(h)) 
        tempM = np.add(A, nv_matrix)
        B = np.linalg.inv(tempM)
        eq_gain_valid_res[:, re] = np.clip(
                                     np.real(
                                        np.diag(
                                            np.matmul(B, A)
                                        )
                                     ), -1e3, 0.9999
                                   )

        BH = np.matmul(B, np.conjugate(h))
        r = rx[:, re]
        eq_outp_valid_res[:, re] = np.matmul(BH, r)

    return (eq_gain_valid_res, eq_outp_valid_res)

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

    hx_shape = (nsym, ntx, nrx, nre)
    rx_shape = (nsym, nrx, nre)
    nv_shape = (ntx, 2)
    
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(sl_mmse_eq_slot_leaf, [(
                H[slot * hx_slot_len : (slot + 1) * hx_slot_len].reshape(hx_shape),
                rx_data[slot * rx_slot_len : (slot + 1) * rx_slot_len].reshape(rx_shape),
                noise_var[slot * nv_slot_len : (slot + 1) * nv_slot_len].reshape(nv_shape)[0][1] * np.eye(ntx,dtype=np.csingle),
                slot,
                valid_res,
                data_syms) for slot in range(nslots)])

    for i, result in enumerate(results):
        eq_gain_all = np.append(eq_gain_all, result[0])
        eq_outp_all = np.append(eq_outp_all, result[1])

    '''
    for slot in range(nslots):
        hx = H[slot * hx_slot_len : (slot + 1) * hx_slot_len].reshape(hx_shape)
        rx = rx_data[slot * rx_slot_len : (slot + 1) * rx_slot_len].reshape(rx_shape)
        nv_matrix = noise_var[slot * nv_slot_len : (slot + 1) * nv_slot_len].reshape(nv_shape)[0][1] * np.eye(ntx,
                dtype=np.csingle)

        eq_gain_slot, eq_outp_slot = sl_mmse_eq_slot_leaf((hx, rx, nv_matrix, slot, valid_res, data_syms))

        eq_gain_all = np.append(eq_gain_all, eq_gain_slot)
        eq_outp_all = np.append(eq_outp_all, eq_outp_slot)
    '''

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
