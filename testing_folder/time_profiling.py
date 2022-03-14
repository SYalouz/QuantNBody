import Quant_NBody
import sys
import pybind.Quant_NBody_fast as Quant_NBody_fast
import testing_folder.Quant_NBody_main_branch as Quant_NBody_old  # This is the original library that I compare with.

from datetime import datetime


def build_a_dagger_a_nbasis_new(n_mo, n_electrons):
    nbody_basis_new = Quant_NBody.build_nbody_basis(n_mo, n_electrons)
    a_dagger_a_new = Quant_NBody.build_operator_a_dagger_a(nbody_basis_new, True)
    return a_dagger_a_new


def build_a_dagger_a_nbasis_fast(n_mo, n_electrons):
    nbody_basis_fast = Quant_NBody_fast.build_nbody_basis(n_mo, n_electrons)
    a_dagger_a_fast = Quant_NBody_fast.build_operator_a_dagger_a_v3(nbody_basis_fast, True)
    return a_dagger_a_fast


def build_a_dagger_a_nbasis_old(n_mo, n_electrons):
    nbody_basis_old = Quant_NBody_old.Build_NBody_Basis(n_mo, n_electrons)
    a_dagger_a_old = Quant_NBody_old.Build_operator_a_dagger_a(nbody_basis_old)
    return a_dagger_a_old


def compare_time1(n_mo, n_electrons, iternum=10):
    build_a_dagger_a_nbasis_old(n_mo, n_electrons)


    a = datetime.now()
    for i in range(iternum):
        build_a_dagger_a_nbasis_old(n_mo, n_electrons)
    time_old = datetime.now() - a
    print(f'For old version it took {datetime.now() - a}')


    a = datetime.now()
    for i in range(iternum):
        print(i)
        build_a_dagger_a_nbasis_new(n_mo, n_electrons)
    time_new = datetime.now() - a
    print(f'For new version it took {datetime.now() - a}')

    a = datetime.now()
    for i in range(iternum):
        print(i)
        build_a_dagger_a_nbasis_fast(n_mo, n_electrons)
    print(f'For fast version it took {datetime.now() - a}')
    time_fast = datetime.now() - a

    return time_old, time_new, time_fast


if __name__ == '__main__':
    pass
    # compare_time1(5, 5, 5)
    # # For old version it took 0:00:08.526233
    # # For new version it took 0:00:06.359180

    # list_time = []
    # for i in range(1, 7):
    #     old, new = compare_time1(8, i, 5)
    #     list_time.append([i, old, new])
    # print(list_time)
    # # [[1, datetime.timedelta(seconds=4, microseconds=800825), datetime.timedelta(microseconds=91214)], [2, datetime.timedelta(seconds=4, microseconds=965682), datetime.timedelta(seconds=1, microseconds=68377)], [3, datetime.timedelta(seconds=7, microseconds=648410), datetime.timedelta(seconds=5, microseconds=472493)], [4, datetime.timedelta(seconds=12, microseconds=855377), datetime.timedelta(seconds=20, microseconds=641449)], [5, datetime.timedelta(seconds=23, microseconds=163650), datetime.timedelta(seconds=50, microseconds=602793)], [6, datetime.timedelta(seconds=38, microseconds=455900), datetime.timedelta(seconds=99, microseconds=615821)]]
    # We see that njit is useful when we have more than 4 electrons njit takes around 4 seconds to start

    # compare_time1(8, 8, 1)

    build_a_dagger_a_nbasis_fast(8, 8)
