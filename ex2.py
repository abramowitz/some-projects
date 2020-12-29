ids = ['205552599', '322621921']
import numpy as np
from pysat.solvers import Solver, Minisat22

def solve_problem(input):
    observations = input['observations']
    queries = input['queries']
    N = len(observations)
    rows_num = len(observations[0])
    cols_num = len(observations[0][0])
    police_num = input['police']
    medics_num = input['medics']
    formula = []
    count = 1

    U_arr = np.arange(count, count +  rows_num* cols_num).reshape((rows_num, cols_num)).astype(int)
    count = count + rows_num * cols_num
    H_arr = np.arange(count, count + rows_num * cols_num * N).reshape((rows_num, cols_num, N)).astype(int)
    count += rows_num * cols_num * N
    if police_num:
        Q_old_arr = np.arange(count, count + rows_num * cols_num * N * 2).reshape((rows_num, cols_num, N,2)).astype(int)
        count += rows_num * cols_num * N * 2
    if medics_num:
        I_old_arr = np.arange(count, count + rows_num * cols_num * N).reshape((rows_num, cols_num, N)).astype(int)
        count += rows_num * cols_num * N
    S_arr = np.arange(count, count + rows_num * cols_num * N * 3).reshape((rows_num, cols_num, N, 3)).astype(int)
    count += rows_num * cols_num * N * 3
    post_act_S_arr = np.arange(count, count + rows_num * cols_num * (N-1) * 3).reshape((rows_num, cols_num, (N-1), 3)).astype(int)
    count += rows_num * cols_num * (N-1) * 3
    post_act_H_arr = np.arange(count, count + rows_num * cols_num * (N-1)).reshape((rows_num, cols_num, (N-1))).astype(int)
    count += rows_num * cols_num * (N-1)
    post_act_I_new_arr = np.arange(count, count + rows_num * cols_num * (N-1) * medics_num).reshape((rows_num, cols_num, (N-1), medics_num)).astype(int)
    count += rows_num * cols_num * (N-1) * medics_num
    post_act_Q_new_arr = np.arange(count, count + rows_num * cols_num * (N-1) * police_num).reshape((rows_num, cols_num, (N-1), police_num)).astype(int)
    count += rows_num * cols_num * (N-1) * police_num
    post_act_I_old_arr = np.arange(count, count + rows_num * cols_num * (N-1)).reshape((rows_num, cols_num, (N-1))).astype(int)  # todo
    count += rows_num * cols_num * (N-1)
    post_act_Q_old_arr = np.arange(count, count + rows_num * cols_num * (N-1) * 2).reshape((rows_num, cols_num, (N-1), 2)).astype(int)  # todo
    count += rows_num * cols_num * (N-1) * 2
    # put your solution here, remember the format needed


    """at least one health condition is true in each i,j,n"""
    for i in range(rows_num):
            for j in range(cols_num):
                for n in range(N):
                    clause = []
                    if police_num:
                        clause += [Q_old_arr[i][j][n][0].item(),Q_old_arr[i][j][n][1].item()]
                    if medics_num:
                        clause += [I_old_arr[i][j][n].item()]
                    formula += [[S_arr[i][j][n][0].item(),S_arr[i][j][n][1].item(),S_arr[i][j][n][2].item(), \
                                 H_arr[i][j][n].item(),U_arr[i][j].item()] + clause]
    """at least one health condition is true in each i,j,n after action"""
    for i in range(rows_num):
        for j in range(cols_num):
            for n in range((N-1)):
                clause = []
                if police_num:
                    clause += [post_act_Q_old_arr[i][j][n][0].item(),  post_act_Q_old_arr[i][j][n][1].item()]
                if medics_num:
                    clause += [post_act_I_old_arr[i][j][n].item()]
                formula += [[post_act_S_arr[i][j][n][0].item(),post_act_S_arr[i][j][n][1].item(),post_act_S_arr[i][j][n][2].item(), \
                                 post_act_H_arr[i][j][n].item(),U_arr[i][j].item()]\
                            +[post_act_Q_new_arr[i][j][n][k].item() for k in range(police_num)]
                            +[post_act_I_new_arr[i][j][n][k].item() for k in range(medics_num)]+ clause]

    """only one health condition is true in each i,j,n
    2 teams are *not* allowed"""
    for i in range(rows_num):
        for j in range(cols_num):
            for n in range(N):
                health_conditions_list = [S_arr[i][j][n][0],S_arr[i][j][n][1],S_arr[i][j][n][2], \
                 H_arr[i][j][n],U_arr[i][j]]
                if police_num:
                    health_conditions_list += [Q_old_arr[i][j][n][0],Q_old_arr[i][j][n][1]]
                if medics_num:
                    health_conditions_list += [I_old_arr[i][j][n]]

                for i1 in range(len(health_conditions_list)):
                    for i2 in range(i1+1,len(health_conditions_list)):
                        formula += [[-health_conditions_list[i1].item(),-health_conditions_list[i2].item()]]
            for n in range(N-1):
                post_act_health_conditions_list = [post_act_S_arr[i][j][n][0],post_act_S_arr[i][j][n][1],post_act_S_arr[i][j][n][2], \
                     post_act_H_arr[i][j][n],U_arr[i][j]]
                if police_num:
                    post_act_health_conditions_list += [post_act_Q_old_arr[i][j][n][0], post_act_Q_old_arr[i][j][n][1]]
                if medics_num:
                    post_act_health_conditions_list += [post_act_I_old_arr[i][j][n]]
                Q_new_list = [post_act_Q_new_arr[i][j][n][k] for k in range(police_num)]
                I_new_list = [post_act_I_new_arr[i][j][n][k] for k in range(medics_num)]
                for i1 in range(len(post_act_health_conditions_list)):
                    for i2 in range(i1 + 1, len(post_act_health_conditions_list)):
                        formula += [[-post_act_health_conditions_list[i1].item(), -post_act_health_conditions_list[i2].item()]]
                    for police_team in Q_new_list:
                        formula += [[-post_act_health_conditions_list[i1].item(), -police_team.item()]]
                    for medic_team in I_new_list:
                        formula += [[-post_act_health_conditions_list[i1].item(), -medic_team.item()]]
                # for police_team in Q_new_list:
                #     for medic_team in I_new_list:
                #         formula += [[-police_team.item(), -medic_team.item()]]
                """only 1 team in a location!!"""
                for i1 in range(len(Q_new_list)):
                    for i2 in range(i1+1,len(Q_new_list)):
                        formula += [[-Q_new_list[i1].item(), -Q_new_list[i2].item()]]
                for i1 in range(len(I_new_list)):
                    for i2 in range(i1+1,len(I_new_list)):
                        formula += [[-I_new_list[i1].item(), -I_new_list[i2].item()]]
    """post action H -> all teams are working!!!!!
    i.e. post_action_H[i,j,n] ->(V{Q_new[i2,j2,n,k]})"""
    for n in range((N-1)):
        for i in range(rows_num):
            for j in range(cols_num):
                formula += [[-post_act_H_arr[i,j,n].item()]
                            +[post_act_I_new_arr[i2,j2,n,k].item() for i2 in range(rows_num) for j2 in range(cols_num)]
                            for k in range(medics_num)]
                formula += [[-post_act_S_arr[i, j, n,r].item()]
                            + [post_act_Q_new_arr[i2, j2,n,k].item() for i2 in range(rows_num) for j2 in range(cols_num)]
                            for k in range(police_num) for r in range(3)]



    """each time, one team in one lacation at most"""
    for n in range((N-1)):
        for k in range(medics_num):
            matrix_in_time_n = post_act_I_new_arr[:,:,n,k].reshape((rows_num*cols_num)).tolist()
            for loc in range(len(matrix_in_time_n)):
                for other_loc in range(loc+1,len(matrix_in_time_n)):
                    formula += [[-matrix_in_time_n[loc],-matrix_in_time_n[other_loc]]]
            """new I -> H before action"""
            for i in range(rows_num):
                for j in range(cols_num):
                    formula += [[-post_act_I_new_arr[i,j,n,k].item(), H_arr[i,j,n].item()]]
        for k in range(police_num):
            matrix_in_time_n = post_act_Q_new_arr[:, :, n, k].reshape((rows_num * cols_num)).tolist()
            for loc in range(len(matrix_in_time_n)):
                for other_loc in range(loc+1,len(matrix_in_time_n)):
                    formula += [[-matrix_in_time_n[loc],-matrix_in_time_n[other_loc]]]
            """new Q -> S before action"""
            for i in range(rows_num):
                for j in range(cols_num):
                    formula += [[-post_act_Q_new_arr[i,j,n,k].item(), S_arr[i,j,n,0].item(),S_arr[i,j,n,1].item(),S_arr[i,j,n,2].item()]]


    for i in range(rows_num):
        for j in range(cols_num):
            for n in range((N-1)):
                """post action S -> S before action"""
                formula += [[S_arr[i,j,n,r].item(),-post_act_S_arr[i,j,n,r].item()] for r in range(3)]
                """post action H -> H before action"""
                formula += [[H_arr[i, j, n].item(), -post_act_H_arr[i, j, n].item()]]
                """post action I old (<)-> old I before action"""
                if medics_num:
                    formula += [[I_old_arr[i,j,n].item(),-post_act_I_old_arr[i,j,n].item()]]
                # formula += [[-I_old_arr[i,j,n].item(),post_act_I_old_arr[i,j,n].item()]]
                """post action Q (<)-> Q before action"""
                if police_num:
                    formula += [[Q_old_arr[i,j,n,r].item(),-post_act_Q_old_arr[i,j,n,r].item()] for r in range(2)]
                # formula += [[-Q_old_arr[i,j,n,r].item(),Q_old_arr[i,j,n,r].item()] for r in range(2)]


    # results
    for n in range(N-1):
        for i in range(rows_num):
            for j in range(cols_num):
                """S and H neighbor -> S neighbor after result"""
                if i>0:
                    formula += [[-post_act_S_arr[i,j,n,r].item(),-post_act_H_arr[i-1,j,n].item(),S_arr[i-1,j,n+1,0].item()] for r in range(3)]
                if j>0:
                    formula += [[-post_act_S_arr[i,j,n,r].item(),-post_act_H_arr[i,j-1,n].item(),S_arr[i,j-1,n+1,0].item()] for r in range(3)]
                if i < rows_num - 1:
                    formula += [[-post_act_S_arr[i, j, n, r].item(), -post_act_H_arr[i + 1, j, n].item(), S_arr[i + 1, j, n + 1, 0].item()] for r in
                                range(3)]
                if j < cols_num - 1:
                    formula += [[-post_act_S_arr[i, j, n, r].item(), -post_act_H_arr[i, j + 1, n].item(), S_arr[i, j + 1, n + 1, 0].item()] for r in
                                range(3)]
                """other direction"""
                formula += [[post_act_H_arr[i,j,n].item(),-S_arr[i,j,n+1,0].item()]]
                other_direction_clause = [-S_arr[i,j,n+1,0].item()]
                if i>0:
                    other_direction_clause += [post_act_S_arr[i-1,j,n,r].item() for r in range(3)]
                if j>0:
                    other_direction_clause += [post_act_S_arr[i, j-1, n, r].item() for r in range(3)]
                if i < rows_num - 1:
                    other_direction_clause += [post_act_S_arr[i+1, j, n, r].item() for r in range(3)]
                if j < cols_num - 1:
                    other_direction_clause += [post_act_S_arr[i, j+1, n, r].item() for r in range(3)]
                formula.append(other_direction_clause)

                """H and no S neighbors -> H after result"""
                health_remaining_clause = [-post_act_H_arr[i,j,n].item(),H_arr[i,j,n+1].item()]
                if i>0:
                    health_remaining_clause += [post_act_S_arr[i-1,j,n,r].item() for r in range(3)]
                if j>0:
                    health_remaining_clause += [post_act_S_arr[i,j-1,n,r].item() for r in range(3)]
                if i < rows_num - 1:
                    health_remaining_clause += [post_act_S_arr[i+1,j,n,r].item() for r in range(3)]
                if j < cols_num - 1:
                    health_remaining_clause += [post_act_S_arr[i,j+1,n,r].item() for r in range(3)]
                formula.append(health_remaining_clause)
                # note: other direction not necessary

                """day of disease: 0(<)->1 , 1(<)->2"""
                formula += [[-post_act_S_arr[i,j,n,r].item(),S_arr[i,j,n+1,r+1].item()] for r in range(2)]
                # formula += [[post_act_S_arr[i,j,n,r].item(),-S_arr[i,j,n+1,r+1].item()] for r in range(2)]
                """last day of disease -> H"""
                formula += [[-post_act_S_arr[i, j, n, 2].item(), H_arr[i, j, n + 1].item()]]
                """post action Q new -> Q 0"""
                formula += [[-post_act_Q_new_arr[i, j, n, k].item(), Q_old_arr[i, j, n + 1,0].item()] for k in range(police_num)]
                """post action Q 0 (<)-> Q 1"""
                if police_num:
                    formula += [[-post_act_Q_old_arr[i, j, n, 0].item(), Q_old_arr[i, j, n + 1,1].item()]]
                # formula += [[post_act_Q_old_arr[i, j, n, 0].item(), -Q_old_arr[i, j, n + 1, 1].item()]]
                """post action I new -> I old"""
                formula += [[-post_act_I_new_arr[i, j, n, k].item(), I_old_arr[i, j, n + 1].item()] for k in range(medics_num)]
                """post action Q 1 -> H"""
                if police_num:
                    formula += [[-post_act_Q_old_arr[i, j, n, 1].item(), H_arr[i, j, n + 1].item()]]
                """post action I old -> I old"""
                if medics_num:
                    formula += [[-post_act_I_old_arr[i, j, n].item(), I_old_arr[i, j, n + 1].item()]]

    """S_observed[i,j,n]==T means observations[n][i][j] == 'S'"""
    # same with Q
    S_observed = np.arange(count, count + rows_num * cols_num * N ).reshape((rows_num, cols_num, N)).astype(int)
    count += rows_num * cols_num * N
    if police_num:
        Q_observed = np.arange(count, count + rows_num * cols_num * N).reshape((rows_num, cols_num, N)).astype(int) # todo
        count += rows_num * cols_num * N
    """S_observed[i,j,n] -> (S[i,j,n,0] or S[i,j,n,1] or S[i,j,n,2])"""
    for n in range(N):
        for i in range(rows_num):
            for j in range(cols_num):
                formula += [[-S_observed[i,j,n].item()]+[S_arr[i,j,n,r].item() for r in range(3)]]
                formula += [[S_observed[i, j, n].item()] + [-S_arr[i, j, n, r].item()] for r in
                            range(3)]  # other direction!
                if police_num:
                    formula += [[-Q_observed[i, j, n].item()] + [Q_old_arr[i, j, n, r].item() for r in range(2)]]
                    formula += [[Q_observed[i, j, n].item()] + [-Q_old_arr[i, j, n, r].item()] for r in range(2)]  #todo??#[]


    """no Q/I/S1/S2 in first round"""
    for i in range(rows_num):
        for j in range(cols_num):
            formula += [[-S_arr[i, j, 0, r].item()] for r in range(1,3)]
            if police_num:
                formula += [[-Q_old_arr[i, j, 0, r].item()] for r in range(2)]
            # formula += [[-Q_observed[i, j, 0].item()]]
            if medics_num:
                formula += [[-I_old_arr[i, j, 0].item()]]

    """observations"""
    for i in range(rows_num):
        for j in range(cols_num):
            # every observed S in n=0 is in his first day of disease
            U_was_seen = False
            for n in range(N):
                if observations[n][i][j] == '?':
                    continue
                if observations[n][i][j] == 'U':
                    if not U_was_seen:
                        formula.append([U_arr[i, j].item()])
                        U_was_seen = True
                    continue
                letter_to_proposition_id = {'H': H_arr[i, j, n], 'S': S_observed[i, j, n]}
                if police_num:
                    letter_to_proposition_id['Q'] = Q_observed[i, j, n]
                if medics_num:
                    letter_to_proposition_id['I'] = I_old_arr[i, j, n]
                formula.append([letter_to_proposition_id[observations[n][i][j]].item()])

    # print(formula)
    # print(len(formula))
    s = Solver(name='g4',bootstrap_with=formula)
    output = {}
    for q in queries:
        # print(q)
        i = q[0][0]
        j = q[0][1]
        n = q[1]
        if (q[2] == 'Q' and not police_num) or (q[2] == 'I' and not medics_num):
            answer = 'F'
        letter_to_proposition_id = {'H': H_arr[i, j, n], 'S': S_observed[i, j, n], 'U': U_arr[i,j]}
        if police_num:
            letter_to_proposition_id['Q'] = Q_observed[i, j, n]
        if medics_num:
            letter_to_proposition_id['I'] = I_old_arr[i, j, n]
        is_possible = s.solve(assumptions=[letter_to_proposition_id[q[2]].item()])
        opposite_is_possible = s.solve(assumptions=[-letter_to_proposition_id[q[2]].item()])
        # if is_possible and opposite_is_possible:
        #     s.solve()
        #     # print(letter_to_proposition_id[q[2]])
        #     for m in s.enum_models():
        #         print(m)
        #         print(formula)
        #         break
        if is_possible and opposite_is_possible:
            answer = '?'
        elif is_possible and not opposite_is_possible:
            answer = 'T'
        elif not is_possible and opposite_is_possible:
            answer = 'F'
        else:
            s.solve()
            print(formula)
            # [-345, 41, 42], [345, -41],
            # print(Q_old_arr[0, 0, 2, 0])
            print(Q_observed)
            for m in s.enum_models():
                print(m)
                print(formula)
                break
            raise Exception('Bad input or error in the code')
        output[q] = answer
        # print(output)
    return output


