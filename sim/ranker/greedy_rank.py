import sys
import numpy as np
import pandas as pd
from .template_rank import AbstractRanker

TAU_INF = 10000000

class GreedyRanker(AbstractRanker):

    def __init__(self,
                include_S = True,
                tau = TAU_INF):
        self.description = "class for tracing greedy inference of openABM loop"
        self.include_S = include_S
        self.tau = tau
        self.rng = np.random.RandomState(1)
    def init(self, N, T):
        self.contacts = []
        #dummy obs, needed if the first time you add only one element
        self.obs = [(0,-1,0)] 
        self.T = T
        self.N = N
        self.rank_not_zero = np.zeros(T)

        return True

    #def rank(self, t_day, daily_contacts, daily_obs, data):
    def rank(self, t_day, daily_contacts, daily_obs):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''

        for obs in daily_obs:
            self.obs.append(obs)

        while len(self.contacts) > 0:
            if self.contacts[0][2] < t_day - self.tau:
                self.contacts.pop()
            else:
                break

        for (i,j,t,l) in daily_contacts:
            self.contacts.append([i,j,t,l])

        obs_df = pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
        contacts_df = pd.DataFrame(self.contacts, columns=["i", "j", "t", "lambda"])
        if not self.include_S:
            obs_df = obs_df[obs_df.s != 0]
        rank_greedy = run_greedy(obs_df, t_day, contacts_df, self.N, self.rng,tau = self.tau, verbose=False) # just infected
        dict_greedy = dict(rank_greedy)
        self.rank_not_zero[t_day] =  sum([1 for x in rank_greedy if x[1] > 0])
        #data["rank_not_zero"] = self.rank_not_zero
        rank = list(sorted(rank_greedy, key=lambda tup: tup[1], reverse=True))

        return rank



def run_greedy(observ, T, contacts, N, rng, noise = 1e-3, tau = TAU_INF, verbose=True):

    observ = observ[(observ["t_test"] <= T)]
    contacts = contacts[(contacts["t"] <= T) & (contacts["t"] >= T-tau)]

    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I

    # debug
    #idx_I_at_T = observ[(observ['s'] == 1) & (observ['t_test'] == T)].to_numpy()
    #idx_I_assumed = np.setdiff1d(idx_I, idx_I_at_T)

    idx_S_anyT = observ[(observ['s'] == 0) & (observ['t_test'] < T)]['i'] # observed S at time < T
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking

    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,N), idx_all) # these have no contacts -> tail of the ranking


    idx_to_inf = np.setdiff1d(idx_all, idx_I) # nor I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # nor S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # nor R anytime


    maxS = -1 * np.ones(N)
    minR = T * np.ones(N)
    for i, s, t_test in  observ[["i", "s", "t_test"]].to_numpy():
        if s == 0 and t_test < T:
            maxS[i] = max(maxS[i], t_test)
        if s == 2:
            minR[i] = min(minR[i], t_test)
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)

    if verbose:
        print("! Assuming contacts as direct links !", file=sys.stderr)
        print("! Assuming that if i is infected at t < T (and not observed as R), it is still infected at T !", file=sys.stderr)

    Score = dict([(i, 0) for i in range(N)])
    print(f"all contacts: {len(contacts)}")
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]
    print(f"all contacts cut: {len(contacts_cut)}")
    
    for i, j, t in contacts_cut[["i", "j", "t"]].to_numpy():
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += 1
    
    for i in range(0,N):
        if verbose:
            if i % 1000 == 0:
                print("Done... "+ str(i) + "/" + str(N))
        if i in idx_non_obs:
            Score[i] = -1 + rng.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max()
        elif i in idx_S: #at time T
            Score[i] = -1 + rng.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + rng.rand() * noise
    sorted_Score = list(sorted(Score.items(),key=lambda item: item[1], reverse=True))
    return sorted_Score



def run_greedy_weighted(observ, T, contacts, N, noise = 1e-3, verbose=True):

    observ = observ[(observ["t_test"] <= T)]
    contacts = contacts[(contacts["t"] <= T)]

    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I

    # debug
    #idx_I_at_T = observ[(observ['s'] == 1) & (observ['t_test'] == T)].to_numpy()
    #idx_I_assumed = np.setdiff1d(idx_I, idx_I_at_T)

    idx_S_anyT = observ[(observ['s'] == 0) & (observ['t_test'] < T)]['i'] # observed S at time < T
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking

    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,N), idx_all) # these have no contacts -> tail of the ranking


    idx_to_inf = np.setdiff1d(idx_all, idx_I) # nor I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # nor S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # nor R anytime


    maxS = dict()
    minR = dict()
    for i in range(0,N):
        if i in idx_S_anyT:
            maxS[i] = observ[(observ['i'] == i) & (observ['s'] == 0)]['t_test'].max()
        else:
            maxS[i] = -1
        if i in idx_R:
            minR[i] = observ[(observ['i'] == i) & (observ['s'] == 2)]['t_test'].min()
        else:
            minR[i] = T
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)

    if verbose:
        print("! Assuming contacts as direct links !", file=sys.stderr)
        print("! Assuming that if i is infected at t < T (and not observed as R), it is still infected at T !", file=sys.stderr)

    Score = dict([(i, 0) for i in range(N)])
    print(f"all contacts: {len(contacts)}")
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]
    print(f"all contacts cut: {len(contacts_cut)}")
    
    for i, j, t, lamb in contacts_cut.to_numpy():
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += lamb
    
    for i in range(0,N):
        if verbose:
            if i % 1000 == 0:
                print("Done... "+ str(i) + "/" + str(N))
        if i in idx_non_obs:
            Score[i] = -1 + np.random.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max()
        elif i in idx_S: #at time T
            Score[i] = -1 + np.random.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + np.random.rand() * noise
    sorted_Score = list(sorted(Score.items(),key=lambda item: item[1], reverse=True))
    return sorted_Score

