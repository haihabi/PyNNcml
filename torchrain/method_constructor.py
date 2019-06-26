from torchrain.wet_dry import STDWetDry


def statistics_wet_dry(th, step):
    return STDWetDry(th, step)
