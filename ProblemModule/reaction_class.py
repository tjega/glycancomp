class Monod:
    def __init__(self, mu_S, K_S):
        self.mu_S, self.K_S = mu_S, K_S

    def monod(self, S, X):
        if S == 0 or S < 0:
            mon = 0.0
        elif X == 0 or X < 0:
            mon = 0.0
            X = 0.0
        else:
            mon = ((self.mu_S * S) / (self.K_S + S)) * X
        return mon


class Contois:
    def __init__(self, mu_I, K_I):
        self.mu_I, self.K_I = mu_I, K_I

    def contois(self, I, X):
        if X == 0 or X < 0:
            con = 0.0
            X = 0.0
        elif I == 0 or I < 0:
            con = 0
        else:
            con = ((self.mu_I * I) / ((self.K_I * X) + I)) * X
        return con


class Decay:
    def __init__(self, alpha):
        self.alpha = alpha

    def decay(self, X):
        if X < 0.0:
            X = 0.0
            dec = 0.0
        elif X == 0.0:
            dec = 0.0
        else:
            dec = self.alpha * X
        return dec


class Mucprod:
    def __init__(self, lam_max, lam_prod):
        self.lam_max, self.lam_prod = lam_max, lam_prod

    def mucprod(self, I):
        if (self.lam_max == 0) or (self.lam_prod == 0):
            muc = 0
        elif (I/self.lam_max) > 1:
            muc = 0
        else:
            muc = (1 - (I/self.lam_max))*self.lam_prod
        return muc


class Mucprodspec:
    def __init__(self, lam_max, lam_prod, lam_spec):
        self.lam_max, self.lam_prod, self.lam_spec = lam_max, lam_prod, lam_spec

    def mucprodspec(self, G, Z):
        if (self.lam_max == 0) or (self.lam_prod == 0 and self.lam_max == 0):
            muc = 0
        elif (G/self.lam_max) > 1:
            muc = 0
        else:
            muc = (1 - (G/self.lam_max))*(self.lam_prod + self.lam_spec*Z)
        return muc


class Prefcontois:
    def __init__(self, mu_GX, K_GX, omega):
        self.mu_GX, self.K_GX, self.omega = mu_GX, K_GX, omega

    def prefcontois(self, I, G, X):
        # function signature must be in the same order as vars_dict
        if X == 0 or X < 0:
            X = 0
        if G == 0 or G < 0 or X == 0 or X < 0:
            pref_con = 0
        else:
            pref_con = ((self.mu_GX * G) / ((self.K_GX * X) + (self.omega * I) + G)) * X
        return pref_con


class Doublemonod:
    def __init__(self, mu_S, K_S, mu_R, K_R):
        self.mu_S, self.K_S, self.mu_R, self.K_R = mu_S, K_S, mu_R, K_R

    def doublemonod(self, S, R, X):
        if S == 0 or S < 0 or R == 0 or R < 0 or X == 0 or X < 0:
            dmon = 0.0
        if X == 0 or X < 0:
            X = 0.0
        else:
            dmon = ((self.mu_S * S) / (self.K_S + S)) * ((self.mu_R * R) / (self.K_R + R)) * X
        return dmon


class Antmonod:
    def __init__(self, mu_S, K_S, K_A):
        self.mu_S, self.K_S, self.K_A = mu_S, K_S, K_A

    def antmonod(self, S, X, A):
        if A < 0.0:
            A = 0.0
        if S == 0 or S < 0:
            antmon = 0.0
        elif X == 0 or X < 0:
            antmon = 0.0
            X = 0.0
        elif A > 0.0 and X <= 0.0:
            antmon = 0.0
            X = 0.0
        else:
            if A > 0.0 and self.K_A > 0.0:
                denom_A = (self.K_A + A)
                antmon = (((self.mu_S * S) / (self.K_S + S)) * (self.K_A /denom_A)) * X
            elif self.K_A == 0.0:
                antmon = ((self.mu_S * S) / (self.K_S + S)) * X
            else:
                antmon = ((self.mu_S * S) / (self.K_S + S)) * X
        return antmon


class Antdoublemonod:
    def __init__(self, mu_S, K_S, mu_R, K_R, K_A):
        self.mu_S, self.K_S, self.mu_R, self.K_R, self.K_A = mu_S, K_S, mu_R, K_R, K_A

    def antdoublemonod(self, S, R, X, A):
        if A < 0.0:
            A = 0.0
        if S == 0 or S < 0 or R == 0 or R < 0:
            antdmon = 0.0
        elif X == 0 or X < 0:
            antdmon = 0.0
            X = 0.0
        elif A > 0.0 >= X:
            antdmon = 0
            X = 0.0
        else:
            denom = (self.K_A + A)
            ant = self.K_A/denom
            if A > 0.0 and self.K_A > 0.0:
                antdmon = (((self.mu_S * S) / (self.K_S + S)) * ((self.mu_R * R) / (self.K_R + R))) * (
                            ant) * X
            elif self.K_A == 0.0:
                antdmon = (((self.mu_S * S) / (self.K_S + S)) * ((self.mu_R * R) / (self.K_R + R))) * X
            else:
                antdmon = (((self.mu_S * S) / (self.K_S + S)) * ((self.mu_R * R) / (self.K_R + R))) * X
        return antdmon