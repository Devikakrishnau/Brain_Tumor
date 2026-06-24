import os

class BatchAgent:

    def run_folder(self, folder, orch):

        results = []

        for f in os.listdir(folder):

            if f.lower().endswith((".jpg",".png",".jpeg")):

                path = os.path.join(folder,f)
                pred,conf,_ = orch.run_pipeline(path,"")

                results.append((f,pred,conf))

        return results
