import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft


class Ising:
    """
    Simulate the 2D Ising Model on an L x L lattice at temperature T.
    
    Attributes
    ----------
    M : list
        Stores magnetization measurements at different simulation steps.
    E : list
        Stores energy measurements at different simulation steps.
    L : int
        Lattice dimension (L x L).
    T : float
        Temperature.
    """

    def __init__(self, L, T):
        """
        Initialize the Ising model with a given lattice size and temperature.
        
        Parameters
        ----------
        L : int
            Lattice dimension (L x L).
        T : float
            Temperature.
        """
        self.M = []
        self.E = []
        self.L = L
        self.T = T

    def monte_carlo_move(self, spins, L, beta, rule):
        """
        Perform a Monte Carlo sweep over the entire L x L spin lattice.

        Two update rules are available:
            1) "metropolis"
            2) "glauber"

        Parameters
        ----------
        spins : np.ndarray
            2D array of shape (L, L) containing spin states (+1 or -1).
        L : int
            Lattice dimension.
        beta : float
            1 / (k_B * T). Here k_B = 1, so beta = 1 / T.
        rule : str
            Update rule: "metropolis" or "glauber".

        Returns
        -------
        np.ndarray
            The updated spin configuration.
        """
        for _ in range(L):
            for _ in range(L):
                x = np.random.randint(0, L)
                y = np.random.randint(0, L)
                S = spins[x, y]

                # Sum of the four neighbor spins
                neighbors = (spins[(x + 1) % L, y] +
                             spins[(x - 1) % L, y] +
                             spins[x, (y + 1) % L] +
                             spins[x, (y - 1) % L])

                delta_E = 2 * S * neighbors

                if rule == "metropolis":
                    # Metropolis acceptance
                    if delta_E <= 0:
                        S = -S
                    else:
                        if np.random.rand() < np.exp(-delta_E * beta):
                            S = -S

                elif rule == "glauber":
                    # Glauber acceptance
                    prob_flip = 0.5 * (1 - math.tanh(delta_E * beta / 2))
                    if np.random.rand() < prob_flip:
                        S = -S

                spins[x, y] = S

        return spins

    def simulate(self, rule="metropolis"):
        """
        Run the Monte Carlo simulation for the 2D Ising model.

        Parameters
        ----------
        rule : str, optional
            The update rule to use, either "metropolis" or "glauber".
            By default, "metropolis".
        """
        L, T = self.L, self.T
        spins = 2 * np.random.randint(2, size=(L, L)) - 1  # Random initial config
        f = plt.figure(figsize=(15, 15), dpi=80)
        
        # Plot initial configuration
        self.config_plot(f, spins, step=0, subplot_index=1)

        mc_steps = 1001
        # Normalization for extensive variables => to get "intensive" measure
        n_factor = 1.0 / (mc_steps * L * L)
        E_temp = 0.0
        M_temp = 0.0
        beta = 1.0 / T

        for i in range(mc_steps):
            spins = self.monte_carlo_move(spins, L, beta, rule)

            # Record measurements every 10 steps
            if i % 10 == 0:
                En = self.calc_energy(spins, L)
                Mag = self.calc_magnetization(spins)

                E_temp += En
                M_temp += Mag

                self.E.append(n_factor * E_temp)
                self.M.append(n_factor * M_temp)

            # Plot a few snapshots
            if i == 1:
                self.config_plot(f, spins, step=i, subplot_index=2)
            if i == 4:
                self.config_plot(f, spins, step=i, subplot_index=3)
            if i == 32:
                self.config_plot(f, spins, step=i, subplot_index=4)
            if i == 100:
                self.config_plot(f, spins, step=i, subplot_index=5)
            if i == 1000:
                self.config_plot(f, spins, step=i, subplot_index=6)

    def calc_energy(self, spins, L):
        """
        Calculate the total energy of the spin configuration.

        Parameters
        ----------
        spins : np.ndarray
            2D array of spins (+1, -1).
        L : int
            Lattice dimension.

        Returns
        -------
        float
            The total energy of the configuration.
        """
        energy = 0.0
        for i in range(L):
            for j in range(L):
                S = spins[i, j]
                neighbors = (spins[(i + 1) % L, j] +
                             spins[(i - 1) % L, j] +
                             spins[i, (j + 1) % L] +
                             spins[i, (j - 1) % L])
                energy += -neighbors * S
        # Each bond counted twice, and each site had 4 neighbors => divide by 4
        return energy / 4.0

    def calc_magnetization(self, spins):
        """
        Return the magnetization of the spin configuration.

        Parameters
        ----------
        spins : np.ndarray
            2D array of spins.

        Returns
        -------
        float
            The total magnetization (sum of spins / 2).
        """
        return np.sum(spins) / 2.0

    def config_plot(self, fig, spins, step, subplot_index):
        """
        Plot the spin configuration at a given simulation step.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to draw on.
        spins : np.ndarray
            2D spin array.
        step : int
            Current simulation step (MC iteration).
        subplot_index : int
            Position of the subplot in the figure (1 to 6).
        """
        ax = fig.add_subplot(3, 3, subplot_index)
        ax.set_xticks([])
        ax.set_yticks([])
        
        L = spins.shape[0]
        X, Y = np.meshgrid(range(L), range(L))
        cax = ax.pcolormesh(X, Y, spins, cmap=plt.cm.RdBu, shading='auto')
        ax.set_title(f"Time={step}")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)


# -------------------------------------------------------------------------
# Simulation and Analysis
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage:  
    # 1) Create an Ising model object
    # 2) Simulate with both "metropolis" and "glauber"
    model = Ising(L=32, T=3.2691)
    model.simulate()
    model.simulate(rule="glauber")

    # Separate Metropolis & Glauber measurements
    # (First 101 measurements from Metropolis, next 101 from Glauber)
    m_met = model.M[:101]
    m_glau = model.M[101:]

    # FFT analysis (Metropolis)
    yf = rfft(m_met)
    xf = rfftfreq(len(m_met), 1 / 10)
    plt.figure()
    plt.plot(xf, np.abs(yf))
    plt.title("FFT of Metropolis Magnetization (T=3.2691)")
    plt.show()

    tau_met = abs(yf[0]) / np.sqrt(np.mean([val**2 for val in m_met]))
    print(f"tau_met={tau_met}")

    # Inverse FFT
    new_sig = irfft(yf)
    plt.figure()
    plt.plot(new_sig[:1000])
    plt.title("Inverse FFT (Metropolis)")
    plt.show()

    # FFT analysis (Glauber)
    yf = rfft(m_glau)
    xf = rfftfreq(len(m_glau), 1 / 10)
    plt.figure()
    plt.plot(xf, np.abs(yf))
    plt.title("FFT of Glauber Magnetization (T=3.2691)")
    plt.show()

    tau_glau = abs(yf[0]) / np.sqrt(np.mean([val**2 for val in m_glau]))
    print(f"tau_glau={tau_glau}")

    new_sig = irfft(yf)
    plt.figure()
    plt.plot(new_sig[:1000])
    plt.title("Inverse FFT (Glauber)")
    plt.show()

    # Repeat for T = 2.2691 (Close to critical temperature for 2D Ising)
    tc_model = Ising(L=32, T=2.2691)
    tc_model.simulate()
    tc_model.simulate(rule="glauber")

    m_met = tc_model.M[:101]
    m_glau = tc_model.M[101:]

    # FFT analysis (Metropolis @ Tc)
    tc_yf = rfft(m_met)
    tc_xf = rfftfreq(len(m_met), 1 / 10)
    plt.figure()
    plt.plot(tc_xf, np.abs(tc_yf))
    plt.title("FFT of Metropolis Magnetization (T=2.2691)")
    plt.show()

    tc_tau_met = abs(tc_yf[0]) / np.sqrt(np.mean([val**2 for val in m_met]))
    print(f"tc_tau_met={tc_tau_met}")

    new_sig = irfft(tc_yf)
    plt.figure()
    plt.plot(new_sig[:1000])
    plt.title("Inverse FFT (Metropolis, T=2.2691)")
    plt.show()

    # FFT analysis (Glauber @ Tc)
    tc_yf = rfft(m_glau)
    tc_xf = rfftfreq(len(m_glau), 1 / 10)
    plt.figure()
    plt.plot(tc_xf, np.abs(tc_yf))
    plt.title("FFT of Glauber Magnetization (T=2.2691)")
    plt.show()

    # Use a different index for amplitude if desired
    tc_tau_glau = abs(tc_yf[1]) / np.sqrt(np.mean([val**2 for val in m_glau]))
    print(f"tc_tau_glau={tc_tau_glau}")

    new_sig = irfft(tc_yf)
    plt.figure()
    plt.plot(new_sig[:1000])
    plt.title("Inverse FFT (Glauber, T=2.2691)")
    plt.show()
