from_camel_conversion_dict = dict(vLeak='v_leak', vThresh='v_thresh', vSynEx='v_syn_ex', vSynIn='v_syn_in', vUnused_4='v_unused_4', vUnused_5='v_unused_5', vUnused_6='v_unused_6', vUnused_7='v_unused_7', vUnused_8='v_unused_8', iBiasSpkCmp='i_bias_spk_cmp', iBiasDelay='i_bias_delay', iBiasLeak='i_bias_leak', iBiasLeakSd='i_bias_leak_sd', iBiasReadOut='i_bias_readout', iRefr='i_refr', iBiasSynGmEx='i_bias_syn_gm_ex', iBiasSynSdEx='i_bias_syn_sd_ex', iBiasSynResEx='i_bias_syn_res_ex', iBiasSynOffEx='i_bias_syn_off_ex', iBiasSynResIn='i_bias_syn_res_in', iBiasSynGmIn='i_bias_syn_gm_in', iUnused_21='i_unused_21', iBiasSynSdIn='i_bias_syn_sd_in', iBiasSynOffIn='i_bias_syn_off_in', vReset='v_reset')

conversion_dict = dict()
for k, v in from_camel_conversion_dict.items():
    conversion_dict[v] = k


def convert_from_camel_case(dictionary):
    new_dict = dict()
    for k, v in dictionary.items():
        if k == 'vReset':
            continue
        new_dict[from_camel_conversion_dict[k]] = v
    return new_dict
