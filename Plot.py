import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


class Plot:
    def __init__(self):
        self.name = "Plot"

    # colour map: agent properties in time
    def Fig1(self, sellerP, numB, capital):
        if sellerP != 0:
            fig = plt.figure()
            # plt.imshow(sellerP, cmap='turbo')
            plt.pcolormesh(sellerP, cmap='jet')
            """boundaries = np.arange(0, p_max, .1)
            cmap_reds = plt.cm.get_cmap('turbo', len(boundaries))
            colors = list(cmap_reds(np.arange(len(boundaries))))
            for i in range(0, 10):
                colors[i] = "white"
            cmap = matplotlib.colors.ListedColormap(colors[:-1], "")
            cmap.set_over(colors[-1])

            cm = plt.pcolormesh(sellerP, cmap=cmap,
                                norm=matplotlib.colors.BoundaryNorm(boundaries, ncolors=len(boundaries) - 1, clip=False))
            """
            plt.title("Prices of seller sites in time")
            plt.ylabel("Time")
            plt.xlabel("Position")
            color_bar = plt.colorbar(label="Price")
            color_bar.minorticks_on()
            fig.savefig("data/Prices_in_time.pdf")
            fig.show()

        if numB != 0:
            fig = plt.figure()
            plt.imshow(numB, cmap='turbo', vmin=0, vmax=np.amax(numB))
            plt.title("Number of buyers at buyer sites in time")
            plt.ylabel("Time")
            plt.xlabel("Position")
            plt.colorbar(label="Counts")
            fig.savefig("data/NumB_in_time.pdf")
            fig.show()

        if capital != 0:
            fig = plt.figure()
            plt.imshow(capital, cmap='turbo')
            plt.title("Capital of seller sites in time")
            plt.ylabel("Time")
            plt.xlabel("Position")
            color_bar = plt.colorbar(label="Capital")
            color_bar.minorticks_on()
            fig.savefig("data/Capital_in_time.pdf")
            fig.show()

    def Fig2(self, sellerP, num_sellers, time_steps, p_max, gamma, beta, delta, seed, num_bins):
        fig = plt.figure()
        widths = [1]
        heights = [1, 3, 3]
        gs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, hspace=1.0)
        init = fig.add_subplot(gs[1, 0])
        final = fig.add_subplot(gs[2, 0])
        plots = [init, final]
        init.hist(sellerP[0], alpha=1, bins=num_bins, color='navy', linewidth=0.8)
        final.hist(sellerP[-1], alpha=1, bins=num_bins, color='navy', linewidth=0.8)

        for plot in plots:
            plot.set_xlim(1, p_max)
            plot.grid(axis='x')
            plot.set_ylabel('Number of sellers')
            plot.set_xlabel('Price')
        init.text(0.01, 1.1, 'Initial distribution', fontsize=10, transform=init.transAxes)
        final.text(0.01, 1.1, 'Final distribution', fontsize=10, transform=final.transAxes)

        title = fig.add_subplot(gs[0, 0])
        title.axis('off')
        t = "Price distribution"

        st = r'Number of sellers: ' + str(num_sellers) + r'; Time steps: ' + str(time_steps) + '\n' + \
             r'$P_{max}$ = ' + str(p_max) + r'; $\gamma$ = ' + str(gamma) + r'; $\beta$ = ' + str(beta) + r'; $\Delta$ = ' + \
             str(delta) + r'; seed = ' + str(seed)

        title.text(0, 1.8, t, fontweight='bold',
                   fontsize=18,
                   verticalalignment='top',
                   horizontalalignment='left')

        title.text(0, 1., st, fontsize=14, verticalalignment='top', horizontalalignment='left')
        fig.savefig("data/Price_distribution.pdf")
        fig.show()

    def Fig3(self, frac_live_bRS, frac_live_aRS, left, bottom, width, height):
        fig = plt.figure()
        plt.title("Fraction of live sellers in time")
        plt.ylabel("Fraction of live sellers (before rebirth)")
        plt.xlabel("Time")
        plt.plot(frac_live_bRS)
        plt.axes([left, bottom, width, height])
        plt.plot(frac_live_aRS, color='orange')
        plt.title('after rebirth')
        fig.savefig("data/Frac_liveS_in_time.pdf")
        fig.show()

    def Fig4(self, t, vacancy_befRebS, vacancy_afterRebS, left, bottom, width, height):
        vac_site_befReb = [0 for i in range(0, t + 1)]
        vac_site_afterReb = [0 for i in range(0, t + 1)]

        for i in range(0, t + 1):
            vac_site_befReb[i] = vacancy_befRebS[i].count(1)
            vac_site_afterReb[i] = vacancy_afterRebS[i].count(1)

        time = np.arange(t + 1)

        fig = plt.figure()
        plt.title("Number of vacant sites in time")
        plt.xlabel("Time")
        plt.ylabel("Number of vacant sites (before rebirth)")
        plt.plot(time, vac_site_befReb)
        plt.axes([left, bottom, width, height])
        plt.plot(time, vac_site_afterReb, color='orange')
        plt.title('after rebirth')
        fig.savefig("data/VacancyS_in_time.pdf")
        fig.show()
