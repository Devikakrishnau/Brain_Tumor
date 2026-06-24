class VotingAgent:

    def vote(self, preds):

        counts = {}
        for p in preds:
            counts[p] = counts.get(p,0)+1

        return max(counts, key=counts.get)
