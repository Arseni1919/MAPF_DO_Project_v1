import time

from functions_plotting import *


def show_results(file_dir):
    plt.close()
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        logs_dict = json.load(openfile)
        expr_type = logs_dict['expr_type']

        if expr_type == 'MAPF':
            logs_dict['alg_names'] = [
                "MACGA+PIBT",
                "MACGA",
                "LaCAM*",
                "LaCAM",
                "PIBT",
                "LNS2",
                "PrP",
            ]

            # fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            #
            # # plot_rsoc(ax, info=logs_dict)
            #
            # plot_sr(ax[0, 0], info=logs_dict)
            # plot_time_metric(ax[0, 1], info=logs_dict)
            # # plot_time_metric_cactus(ax[0, 1], info=logs_dict)
            # # plot_makespan(ax[1, 0], info=logs_dict)
            # plot_makespan_cactus(ax[1, 0], info=logs_dict)
            # # plot_soc(ax[1, 1], info=logs_dict)
            # plot_soc_cactus(ax[1, 1], info=logs_dict)
            # # plot_sr(ax, info=logs_dict)
            # # plot_time_metric_cactus(ax, info=logs_dict)

            sleep_t = 4
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # plot_sr(ax, info=logs_dict)
            # plot_time_metric_cactus(ax, info=logs_dict)
            plot_makespan_cactus(ax, info=logs_dict)
            # plot_time_metric(ax, info=logs_dict)
            # plot_makespan(ax, info=logs_dict)
            # plot_soc(ax, info=logs_dict)
            # plot_soc_cactus(ax, info=logs_dict)

        if expr_type == 'LMAPF':
            # logs_dict['alg_names'] = [
            #     "L-LNS2-A*",
            #     "APF-L-LNS2-A*",
            #     "L-LNS2-SIPPS",
            #     "APF-L-LNS2-SIPPS",
            #     "L-PIBT",
            #     "L-PrP-A*",
            #     "APF-L-PrP-A*",
            #     "L-PrP-SIPPS",
            #     "APF-L-PrP-SIPPS",
            #     # "APF-L-PIBT",
            #     "L-LaCAM",
            #     'L-LaCAM*'
            # ]
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            plot_throughput(ax, info=logs_dict)

        plt.tight_layout()
        plt.show()


def main():

    # file_dir = 'MAPF_2024-09-04--19-12_ALGS-6_RUNS-3_MAP-maze-32-32-2.json'

    # LMAPF
    # file_dir = 'LMAPF_2024-09-21--12-03_ALGS-3_RUNS-5_MAP-empty-32-32.json'

    # MAPF
    # ICAPS-25
    # file_dir = 'MAPF_2024-09-04--21-51_ALGS-6_RUNS-15_MAP-empty-32-32.json'
    # file_dir = 'MAPF_2024-09-05--09-46_ALGS-6_RUNS-15_MAP-random-32-32-10.json'
    # file_dir = 'MAPF_2024-09-05--17-52_ALGS-6_RUNS-15_MAP-random-32-32-20.json'
    # file_dir = 'MAPF_2024-09-05--20-55_ALGS-6_RUNS-15_MAP-room-32-32-4.json'
    # file_dir = 'MAPF_2024-09-06--11-44_ALGS-6_RUNS-15_MAP-maze-32-32-2.json'
    # file_dir = 'MAPF_2024-09-07--16-36_ALGS-6_RUNS-15_MAP-maze-32-32-4.json'

    # IJCAI-25
    # file_dir = 'MAPF_2025-01-19--13-50_ALGS-7_RUNS-20_MAP-empty-32-32.json'
    # file_dir = 'MAPF_2025-01-19--16-02_ALGS-7_RUNS-20_MAP-random-32-32-10.json'
    # file_dir = 'MAPF_2025-01-19--18-18_ALGS-7_RUNS-20_MAP-random-32-32-20.json'
    # file_dir = 'MAPF_2025-01-19--23-27_ALGS-7_RUNS-20_MAP-maze-32-32-4.json'
    # file_dir = 'MAPF_2025-01-20--02-32_ALGS-7_RUNS-20_MAP-maze-32-32-2.json'
    file_dir = 'MAPF_2025-01-20--10-24_ALGS-7_RUNS-20_MAP-room-32-32-4.json'

    # file_dir = 'MAPF_2025-01-20--11-40_ALGS-2_RUNS-20_MAP-random-32-32-20.json'
    # file_dir = 'MAPF_2025-01-20--11-52_ALGS-1_RUNS-20_MAP-random-32-32-20.json'
    # file_dir = 'MAPF_2025-01-20--12-53_ALGS-2_RUNS-20_MAP-room-32-32-4.json'

    # LMAPF - CGA Extension paper
    # file_dir = 'LMAPF_2024-12-28--12-30_ALGS-2_RUNS-25_MAP-empty-32-32.json'
    # file_dir = 'LMAPF_2024-12-28--12-49_ALGS-2_RUNS-25_MAP-random-32-32-20.json'
    # file_dir = 'LMAPF_2024-12-28--12-56_ALGS-2_RUNS-25_MAP-maze-32-32-4.json'
    # file_dir = 'LMAPF_2024-12-28--13-14_ALGS-2_RUNS-25_MAP-room-32-32-4.json'



    # parameters
    # file_dir = ''

    # show_results(file_dir=f'logs_for_experiments/{file_dir}')
    show_results(file_dir=f'final_logs_CGA_MAPF_Paper/{file_dir}')
    # show_results(file_dir=f'final_logs_CGA_Extension_Paper/{file_dir}')


if __name__ == '__main__':
    main()
