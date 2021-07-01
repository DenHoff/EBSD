import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


from Buyer import Buyer
from Seller import Seller

if __name__ == '__main__':

    # --------   INPUT VARIABLES   --------
    num_sellers: int
    num_sellers = 1000
    time_steps: int
    time_steps = 250
    p_max: float
    p_max = 4
    gamma: float
    gamma = 0.75
    delta: float
    delta = 0.04
    seed: int
    seed = 42

    # --------   CREATE DATASETS   --------

    capitalS = [[0 for i in range(num_sellers)] for j in range(time_steps + 1)]
    priceS = [[0 for i in range(num_sellers)] for j in range(time_steps + 1)]
    vacancy_befRebS = [[0 for i in range(num_sellers)] for j in range(time_steps + 1)]
    vacancy_afterRebS = [[0 for i in range(num_sellers)] for j in range(time_steps + 1)]
    num_buyersB = [[0 for i in range(num_sellers)] for j in range(time_steps + 1)]

    frac_live_bRS = [0 for i in range(time_steps + 1)]
    frac_live_bRS[0] = 1
    frac_live_aRS = [0 for i in range(time_steps + 1)]
    frac_live_aRS[0] = 1

    # --------   CREATE NETWORK & INITIALISE   --------
    n: int
    n = 2 * num_sellers
    network = []

    # ensure reproducibility of results by choosing a certain seed
    # interesting seeds: 10, 42, 50
    random.seed(seed)

    step: int
    step = 0
    step_b: int
    step_b = 0
    for x in range(0, n):
        # sellers at even indices
        if x % 2 == 0:
            s = Seller(0, 0, 0)
            s.set_capital(0)
            s.set_fixed_price(random.uniform(1.0, p_max))
            s.set_vacant(0)
            network.append(s)
            capitalS[0][step] = network[x].get_capital()
            priceS[0][step] = network[x].get_fixed_price()
            vacancy_befRebS[0][step] = network[x].get_vacant()
            vacancy_afterRebS[0][step] = network[x].get_vacant()
            step += 1
        # buyers at odd indices
        else:
            b = Buyer(1)
            b.set_num_buyers(1)
            network.append(b)
            num_buyersB[0][step_b] = network[x].get_num_buyers()
            step_b += 1

    # --------   DYNAMICS   --------

    for i in range(0, time_steps):
        print("TIME-STEP " + str(i))

        # N iterations of randomly selecting 1 buyer and 1 seller
        for j in range(0, num_sellers):
            pos = random.randint(0, n-1)
            if pos % 2 == 0:
                random_s = pos
                while pos % 2 != 1:
                    pos = random.randint(0, n-1)
                random_b = pos
            else:
                random_b = pos
                while pos % 2 != 0:
                    pos = random.randint(0, n-1)
                random_s = pos

            # randomly chosen seller pays overhead of 2
            if network[random_s].get_vacant() == 0:
                network[random_s].set_capital(network[random_s].get_capital() - 2)

            # randomly chosen buyer purchases from cheaper adjacent seller
            # last buyer in network is connected to seller at index 0
            if random_b == n - 1:
                next_rightS = 0
            else:
                next_rightS = random_b + 1
            # seller at random_b-1 dead - buy from seller random_b+1 if alive
            if network[random_b - 1].get_vacant() == 1 and network[next_rightS].get_vacant() == 0:
                network[next_rightS].set_capital(network[next_rightS].get_capital() +
                                                 network[next_rightS].get_fixed_price() * network[
                                                     random_b].get_num_buyers())
            # seller at random_b-1 alive - go to cheaper seller
            elif network[random_b - 1].get_vacant() == 0:
                if network[next_rightS].get_vacant() == 1:
                    network[random_b - 1].set_capital(network[random_b - 1].get_capital() +
                                                      network[random_b - 1].get_fixed_price() * network[
                                                          random_b].get_num_buyers())
                elif network[next_rightS].get_vacant() == 0:
                    if network[random_b - 1].get_fixed_price() < network[next_rightS].get_fixed_price():
                        network[random_b - 1].set_capital(network[random_b - 1].get_capital() +
                                                          network[random_b - 1].get_fixed_price() * network[
                                                              random_b].get_num_buyers())
                    elif network[next_rightS].get_fixed_price() < network[random_b - 1].get_fixed_price():
                        network[next_rightS].set_capital(network[next_rightS].get_capital() +
                                                         network[next_rightS].get_fixed_price() * network[
                                                             random_b].get_num_buyers())
                    else:
                        # if two sellers charge same price random purchase decision
                        r: int
                        r = random.choice([random_b - 1, next_rightS])
                        network[r].set_capital(network[r].get_capital() +
                                               network[r].get_fixed_price() * network[random_b].get_num_buyers())

        # sellers with negative capital become bankrupt
        step = 0
        for l in range(0, n, 2):
            if network[l].get_capital() < 0:
                network[l].set_vacant(1)
                network[l].set_fixed_price(0)
                vacancy_befRebS[i+1][step] = network[l].get_vacant()
                # print("seller at " + str(l) + " bankrupt " + str(network[l].get_vacant()))

                # re-population with probability gamma
                if random.uniform(0.0, 1.0) < gamma:
                    network[l].set_vacant(0)
                    network[l].set_capital(0)

                    # copy price from existing seller
                    ex_seller = random.randrange(0, n, 2)
                    while network[ex_seller].get_vacant() == 1:
                        ex_seller = random.randrange(0, n, 2)
                    p_copy: float
                    dp: float
                    p_copy = network[ex_seller].get_fixed_price()
                    dp = random.uniform(-min(delta, p_copy), delta)
                    """while p_copy + dp < 0:
                        ex_seller = random.randrange(0, n, 2)
                        while network[ex_seller].get_vacant() == 1:
                            ex_seller = random.randrange(0, n, 2)
                        p_copy = network[ex_seller].get_fixed_price()
                        dp = random.uniform(-min(delta, p_copy), delta)"""
                    network[l].set_fixed_price(p_copy + dp)
                    # print("new seller at " + str(l) + " has price " + str(network[l].get_fixed_price())
                    #      + " and capital " + str(network[l].get_capital()))

            vacancy_afterRebS[i+1][step] = network[l].get_vacant()
            step += 1

        # --------   STORE DATA IN DATASETS   --------
        step = 0
        for a in range(0, n, 2):
            capitalS[i + 1][step] = network[a].get_capital()
            priceS[i + 1][step] = network[a].get_fixed_price()
            step += 1

        step = 0
        for b in range(1, n + 1, 2):
            num_buyersB[i + 1][step] = network[b].get_num_buyers()
            step += 1

        # vacancy status of sellers' sites
        # print("# sellers vacant - before: " + str(vacancy_befRebS[i+1].count(1)) + "; after: " + str(vacancy_afterRebS[i + 1].count(1)))
        frac_live_bRS[i + 1] = vacancy_befRebS[i + 1].count(0) / num_sellers
        frac_live_aRS[i + 1] = vacancy_afterRebS[i + 1].count(0) / num_sellers
        # print("fraction live sellers (b//a): " + str(frac_live_bRS[i + 1]) + " // " + str(frac_live_aRS[i]))

    print("mean = " + str(np.mean(frac_live_bRS[1:])) + " // " + str(np.mean(frac_live_aRS[1:])))

    # --------   WRITE DATA IN FILES   --------

    import csv

    with open('config.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["num_sellers", "time_steps", "p_max", "gamma", "delta", "seed"])
        writer.writerow([num_sellers, time_steps, p_max, gamma, delta, seed])

    r: int
    r = -1

    with open('Capital.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in capitalS:
            writer.writerow([r, ' '.join([str(a) for a in row])])
            r += 1

    r = -1

    with open('Price.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in priceS:
            writer.writerow([r, ' '.join([str(a) for a in row])])
            r += 1

    r = -1

    with open('Vacancy_bef_reb.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in vacancy_befRebS:
            writer.writerow([r, ' '.join([str(a) for a in row])])
            r += 1

    r = -1

    with open('Vacancy_after_reb.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in vacancy_afterRebS:
            writer.writerow([r, ' '.join([str(a) for a in row])])
            r += 1

    r = -1

    with open('Frac_liveS_bef_reb.csv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["mean", "variance"])
        writer.writerow([np.mean(frac_live_bRS[1:]), np.var(frac_live_bRS[1:])])
        writer.writerow([])
        for column in frac_live_bRS:
            writer.writerow([r, ''.join([str(a) for a in str(column)])])
            r += 1

    # --------   PLOTTING   --------

    # --- Number of sellers vs. price at fixed t ---
    """fig1 = plt.figure()
    widths = [1]
    heights = [1, 3, 3]
    gs = fig1.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths,
                           height_ratios=heights, hspace=1.0)
    init = fig1.add_subplot(gs[1, 0])
    final = fig1.add_subplot(gs[2, 0])
    plots = [init, final]
    init.hist(priceS[0], alpha=1, bins=100, color='navy', linewidth=0.8)
    final.hist(priceS[-1], alpha=1, bins=100, color='navy', linewidth=0.8)

    for plot in plots:
        plot.set_xlim(1, p_max)
        plot.grid(axis='x')
        plot.set_ylabel('Number of sellers')
        plot.set_xlabel('Price')
    init.text(0.01, 1.1, 'Initial distribution', fontsize=10, transform=init.transAxes)
    final.text(0.01, 1.1, 'Final distribution', fontsize=10, transform=final.transAxes)

    title = fig1.add_subplot(gs[0, 0])
    title.axis('off')
    t = "Price distribution"

    st = r'Number of sellers: ' + str(num_sellers) + '\n' + r'Time steps: ' + str(time_steps) + '\n' + \
         r'$P_{max}$ = ' + str(p_max) + r'; $\gamma$ = ' + str(gamma) + r'; $\Delta$ = ' + str(delta) + \
         r'; seed = ' + str(seed)

    title.text(0, 1.8, t, fontweight='bold',
               fontsize=18,
               verticalalignment='top',
               horizontalalignment='left')

    title.text(0, 1., st, fontsize=14, verticalalignment='top', horizontalalignment='left')
    fig1.savefig("Price_distribution.pdf")
    fig1.show()"""

    # --- Vacancy of seller sites in time ---

    """vac_site_befReb = [0 for i in range(0, time_steps)]
    vac_site_afterReb = [0 for i in range(0, time_steps)]

    for i in range(0, time_steps):
        vac_site_befReb[i] = vacancy_befRebS[i+1].count(1)
        vac_site_afterReb[i] = vacancy_afterRebS[i+1].count(1)

    time = np.arange(time_steps)

    fig5 = plt.figure()
    plt.title("Number of vacant sites in time")
    plt.xlabel("Time")
    plt.ylabel("Number of vacant sites (before rebirth)")
    plt.plot(time, vac_site_befReb)
    plt.axes([0.6, 0.2, 0.25, 0.25])  # [left, bottom, width, height]
    plt.plot(time, vac_site_afterReb, color='orange')
    plt.title('after rebirth')
    fig5.savefig("VacancyS_in_time.pdf")
    fig5.show()"""

    # --- Fraction of live seller sites in time ---
    """fig3 = plt.figure()
    plt.title("Fraction of live sellers in time")
    plt.ylabel("Fraction of live sellers (before rebirth)")
    plt.xlabel("Time")
    plt.plot(frac_live_bRS[1:])
    plt.axes([0.6, 0.55, 0.25, 0.25])
    plt.plot(frac_live_aRS[1:], color='orange')
    plt.title('after rebirth')
    fig3.savefig("Frac_liveS_in_time.pdf")
    fig3.show()"""

    seller_price = np.array(priceS.copy())
    seller_price_aR = np.array(priceS.copy())
    seller_capital = np.array(capitalS.copy())

    mean_capital = []
    mean_price = []
    for a in range(0, time_steps + 1):
        deadS = np.where(np.array(vacancy_befRebS[a]) == 1)
        deadS_aR = np.where(np.array(vacancy_afterRebS[a]) == 1)
        for b in range(len(deadS)):
            seller_capital[a, deadS[b]] = np.nan
            seller_price[a, deadS[b]] = np.nan
            seller_price_aR[a, deadS_aR[b]] = np.nan
        mean_capital.append(np.nanmean(seller_capital[a]))
        mean_price.append(np.nanmean(seller_price[a]))

    # --- Prices of sellers in time ---
    fig = plt.figure()
    current_cmap = matplotlib.cm.get_cmap("jet").copy()
    current_cmap.set_bad(color='white')
    plt.pcolormesh(seller_price_aR, cmap=current_cmap, rasterized=True)
    plt.title("Prices of seller sites in time")
    plt.ylabel("Time")
    plt.xlabel("Position")
    plt.colorbar(label="Price")
    fig.savefig("Prices_in_time.pdf")
    fig.show()

    # --- mean properties of live sellers in time ---
    fig4, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    time = np.arange(time_steps)
    #fig4.suptitle('Mean price and capital of live sellers in time')
    ax1.plot(frac_live_bRS[1:])
    ax1.set(ylabel='Fraction alive')
    ax2.plot(time, mean_price[1:])
    ax2.set(ylabel='Mean price')
    ax3.plot(time, mean_capital[1:])
    ax3.set(xlabel='Time', ylabel='Mean capital')
    fig4.savefig("Seller_params_osc.pdf")
    fig4.show()
