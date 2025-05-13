from py_wake import np
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

# Power and Ct curves for SG 11.0-200 DD (94% availability accounted)
power_curve = np.array([
    [4, 280.2],
    [5, 799.1],
    [6, 1532.7],
    [7, 2506.1],
    [8, 3730.7],
    [9, 5311.8],
    [10, 7286.5],
    [11, 9698.3],
    [12, 10639.1],
    [13, 10648.5],
    [14, 10639.3],
    [15, 10683.7],
    [16, 10642],
    [17, 10640],
    [18, 10639.9],
    [19, 10652.8],
    [20, 10646.2],
    [21, 10644],
    [22, 10641.2],
    [23, 10639.5],
    [24, 10643.6],
    [25, 10635.7],
]) * [1, 0.94]  # Apply 6% loss

ct_curve = np.array([
    [4, 0.923],
    [5, 0.919],
    [6, 0.904],
    [7, 0.858],
    [8, 0.814],
    [9, 0.814],
    [10, 0.814],
    [11, 0.814],
    [12, 0.577],
    [13, 0.419],
    [14, 0.323],
    [15, 0.259],
    [16, 0.211],
    [17, 0.175],
    [18, 0.148],
    [19, 0.126],
    [20, 0.109],
    [21, 0.095],
    [22, 0.084],
    [23, 0.074],
    [24, 0.066],
    [25, 0.059],
])

class SG11MW200DD(WindTurbine):
    """
    Siemens Gamesa SG 11.0-200 DD offshore turbine
    Rotor diameter: 200 m
    Hub height: assumed 120 m (adjust based on project data)
    """

    def __init__(self, method='linear'):
        u, p = power_curve.T
        ct = ct_curve[:, 1]
        WindTurbine.__init__(
            self,
            name='SG 11.0-200 DD',
            diameter=200,
            hub_height=120,
            powerCtFunction=PowerCtTabular(u, p * 1000, 'w', ct,
                                           ws_cutin=4, ws_cutout=25,
                                           ct_idle=0.059, method=method)
        )

# Optional: test visualization
def main():
    wt = SG11MW200DD()
    print('Diameter:', wt.diameter())
    print('Hub height:', wt.hub_height())
    
    ws = np.arange(3, 26)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='Power [W]')
    c = plt.plot([], label='Ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    ax.set_ylabel('Ct')
    plt.xlabel('Wind speed [m/s]')
    plt.title('SG 11.0-200 DD Power & Ct curves')
    plt.show()

if __name__ == '__main__':
    main()
