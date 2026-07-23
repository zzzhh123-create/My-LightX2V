CALIB = {"absmax": {}}

# For Wan2.2 MoE: the two experts (high_noise / low_noise) share identical weight
# names (blocks.N....), so a single global CALIB bucket would let one expert
# overwrite the other. These per-expert buckets keep them separate. The active
# expert is set by MultiModelStruct.get_current_model_index during calibration.
CALIB_MOE = {"high_noise": {"absmax": {}}, "low_noise": {"absmax": {}}}
CURRENT_CALIB_EXPERT = {"name": None}
