def refine_load_from_sd(sd):
    for k in list(sd.keys()):
        if 'img_neck.' in k or 'lifter.anchor' in k:
            del sd[k]
    return sd