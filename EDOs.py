
# Implementação de métodos numéricos para resolução de EDOs aplicada à solução da equação de Rayleigh - Plesset para dinâmica de cavitações

import numpy as np
import matplotlib.pylab as plt

# ---- Exemplo de EDO a ser resolvida ------


def f(t, y):
    return -y

# ---- Exemplo de sistema de EDO a ser resolvido -----


def sys(t, r):
    x, y = r
    a = 1.1
    b = 0.4
    c = 0.4
    d = 0.1
    return np.array([a*x - b*x*y, -c*y + d*x*y])

# ---- Equação de Rayleigh-Plesset ------


def rp(t, r):
    radius, u = r

    s = np.float64(0.07287)  # [N/m] tensão superficial da água a 20 graus C
    rho = np.float64(998.2)  # [kg/m^3] densidade da água a 20 graus C
    gamma = np.float64(5/3)  # indice adiabático
    c = np.float64(1481)  # [m/s] velocidade do som na água a 20 graus C
    mi = np.float64(8.9e-4)  # [Pa*s] viscosidade da água a 20 graus C
    Pzero = np.float64(10.1325e5)  # [atm] pressão ambiente (kPa agora)
    Rzero = np.longdouble(2e-6)  # [m] raio estático (equilíbrio)
    # van der waals hard core of gas bubble (argon)
    vdwhc = np.longdouble(Rzero/8.86)
    # [kHz] frequência ultrassônica de ressonância da bolha (depende do meio)
    omega = np.float64(2*np.pi*26500)
    # [atm] pressão que força o colapso das paredes da bolha
    Pa = np.float64(1367900)
    k = 1.33

    # função que define a variação da pressão que força o colapso das paredes da bolha com o tempo
    pT = -Pa*np.sin(omega*t)
    # função que define a pressão do gás contido na bolha que varia com o tempo
    Pgas = np.longdouble(Pzero + (2*s/Rzero))*(((Rzero) **
                                                3 - (vdwhc)**3) / ((radius)**3 - (vdwhc)**3))**gamma

    return np.array([u, ((Pgas - Pzero - pT - 4*mi*(u/radius) - 2*s/radius + ((2*s)/Rzero + Pzero)*(Rzero/radius)**(3*k))*(1/(rho*radius))) - (3/2)*((u)**2/radius)], dtype=np.longdouble)

# ---- Função calcular EDO por Euler -----


def odeEuler(f, y0, t0, NUMBER_OF_STEPS=100, h=0.01):

    y = np.zeros(NUMBER_OF_STEPS, dtype=np.float32)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float32)

    y[0] = y0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):
        K1 = f(t[n], y[n])
        y[n+1] = y[n] + K1*h
        t[n+1] = t[n] + h

    return t, y

# ---- Função calcular EDO por Heun -----


def odeHeun(f, y0, t0, NUMBER_OF_STEPS=100, h=0.01):

    y = np.zeros(NUMBER_OF_STEPS, dtype=np.float32)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float32)

    y[0] = y0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):
        t[n+1] = t[n] + h
        K1 = f(t[n], y[n])
        K2 = f(t[n+1], y[n] + K1*h)
        y[n+1] = y[n] + 0.5*(K1+K2)*h

    return t, y

# ---- Função calcular sistema de EDO por Euler -----


def odeEulerSys(sys, r0, t0, NUMBER_OF_STEPS=100, h=0.01):

    NUMBER_OF_EQUATIONS = len(r0)

    r = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float64)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float64)

    r[0] = r0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):
        t[n+1] = t[n] + h
        K1 = np.array([sys(t[n], r[n])], dtype=np.float64)
        r[n+1] = r[n] + (K1)*h

    return t, r

# ---- Função calcular sistema de EDO por Heun -----


def odeHeunSys(sys, r0, t0, NUMBER_OF_STEPS=100, h=0.01):

    NUMBER_OF_EQUATIONS = len(r0)

    r = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float32)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float32)

    r[0] = r0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):
        t[n+1] = t[n] + h
        K1 = sys(t[n], r[n])
        K2 = sys(t[n+1], r[n] + K1*h)
        r[n+1] = r[n] + 0.5*(K1+K2)*h

    return t, r

# ---- Função calcular sistema de EDO por Runge-Kutta -----


def odeRungeKuttaSys(rp, r0, t0, NUMBER_OF_STEPS=100, h=0.01):

    NUMBER_OF_EQUATIONS = len(r0)

    r = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float64)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float64)

    r[0] = r0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):
        t[n+1] = t[n] + h
        K1 = rp(t[n], r[n])
        K2 = rp(t[n]+0.5*h, r[n] + 0.5*K1*h)
        K3 = rp(t[n]+0.5*h, r[n] + 0.5*K2*h)
        K4 = rp(t[n]+h, r[n] + K3*h)
        r[n+1] = r[n] + (K1+2*K2+2*K3+K4)*(h/6)

        if (np.isnan(np.any(r))):
            r[n+1] = np.float64(0.0)

    return t, r

# ---- Função calcular sistema de EDO por Runge-Kutta-Fehlberg (Adaptativo) ---


def odeRungeKuttaFehlbergSys(sys, r0, t0, NUMBER_OF_STEPS=100, h=0.01, TOL=10E-3):

    NUMBER_OF_EQUATIONS = len(r0)

    r = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float64)
    r4 = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float64)
    r5 = np.zeros([NUMBER_OF_STEPS, NUMBER_OF_EQUATIONS], dtype=np.float64)
    t = np.zeros(NUMBER_OF_STEPS, dtype=np.float64)

    r[0] = r0
    r4[0] = r0
    r5[0] = r0
    t[0] = t0

    for n in range(0, NUMBER_OF_STEPS-1):

        t[n+1] = t[n] + h

        K1 = sys(t[n], r[n])
        K2 = sys(t[n]+(1/4)*h, r[n] + (1/4)*K1*h)
        K3 = sys(t[n]+(3/8)*h, r[n] + ((3*K1 + 9*K2)/32)*h)
        K4 = sys(t[n]+(12/13)*h, r[n] + ((1932*K1 - 7200*K2 + 7296*K3)/2197)*h)
        K5 = sys(t[n]+h, r[n] + ((439/216)*K1 - 8 *
                                 K2 + (3680/513)*K3 - (845/4104)*K4)*h)
        K6 = sys(t[n]+0.5*h, r[n] + ((-8/27)*K1 + 2*K2 -
                                     (3544/2565)*K3 + (1859/4104)*K4 - (11/40)*K5)*h)

        r5[n+1] = r[n] + ((16/135)*K1 + (6656/12825)*K3 +
                          (28561/56430)*K4 - (9/50)*K5 + (2/55)*K6)*h
        r4[n+1] = r[n] + ((25/216)*K1 + (1408/2565)*K3 +
                          (2197/4104)*K4 - (1/5)*K5)*h

        localError = np.float64(r4[n+1] - r5[n+1])

        if (localError[1] != 0.0):
            q = 0.5*(((TOL*h)/abs(localError[1]))**0.25)

        else:
            q = 0.5*(((TOL*h)/abs(1e-25))**0.25)

        if (q < 1):
            h = h*q
            t[n+1] = t[n] + h
            r4[n+1] = r[n] + ((25/216)*K1 + (1408/2565)*K3 +
                              (2197/4104)*K4 - (1/5)*K5)*h
            r[n+1] = r4[n+1]

        else:
            r[n+1] = r4[n+1]
            h = h*q

    return t, r


# ---- Plotagem dos gráficos no pyplot ------

# t, r = odeRungeKuttaFehlbergSys(rp, (2e-6, 0), 0, NUMBER_OF_STEPS=100000, h=1e-9, TOL=10E-7)
t, r = odeRungeKuttaSys(rp, (2e-6, 0), 0, NUMBER_OF_STEPS=100000, h=1e-8)
# t, r = odeEulerSys(rp, (2e-6, 0), 0, NUMBER_OF_STEPS=100000, h=1e-9)
# t, r = odeHeunSys(rp, (2e-6, 0), 0, NUMBER_OF_STEPS=100000, h=1e-9)

# t, r = odeRungeKuttaFehlbergSys(sys, (10, 10), 0, NUMBER_OF_STEPS=10000)
# t, r = odeRungeKuttaSys(sys, (10, 5), 0, NUMBER_OF_STEPS=1000, h=0.05)

radius = r[:, 0]
velocity = r[:, 1]

plt.figure(1)
plt.plot(t, radius, "r")
plt.xlabel('time')
plt.ylabel('radius')

plt.figure(2)
plt.plot(t, velocity, "g")
plt.xlabel('time')
plt.ylabel('velocity')

plt.show()
