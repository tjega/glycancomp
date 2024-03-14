class Diffusion:
    def __init__(self, sign, gam, vol):
        self.sign, self.gam, self.vol = float(sign), float(gam), float(vol)

    def diffusion(self):
        if self.gam == 0.0:
            dif = 0.0
        else:
            dif = self.sign*(self.gam / self.vol)
        return dif


class Exin:
    def __init__(self, gam, vol_this, vol_other):
        self.gam, self.vol_this, self.vol_other = gam, vol_this, vol_other

    def exin(self):
        ex = (self.vol_other/self.vol_this)*self.gam
        if self.gam == 0:
            ex = 0
        return ex


class Exout:
    def __init__(self, gam):
        self.gam = gam

    def exout(self):
        ex = (-1)*self.gam
        if self.gam == 0:
            ex = 0
        return ex
