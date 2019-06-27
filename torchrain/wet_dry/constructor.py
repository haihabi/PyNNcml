from torchrain.wet_dry.std_wd import STDWetDry


def statistics_wet_dry(th, step):
    return STDWetDry(th, step)
