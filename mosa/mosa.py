import json
from copy import deepcopy

try:
    from numpy.random import choice,uniform,shuffle
    from numpy import exp
except:
    from random import choice,uniform,shuffle
    from math import exp

class Anneal:
    def __init__(self):
        '''
        __init__() -> class constructor, initializes object attributes.

        Returns
        -------
        None.
        '''
        initemp=1.0
        decrease=0.9
        ntemp=10
        
        print("--------------------------------------------------")
        print("    MULTI-OBJECTIVE SIMULATED ANNEALING (MOSA)    ")
        print("--------------------------------------------------")
        print("Developed by Dr. Roberto Gomes de Aguiar Veiga")
        print("Universidade Federal do ABC, Brazil")
        print("\n")
                
        self.population={"X":[]}
        self.exchangeprob={}
        self.xnel={}
        self.xdistinct={}
        self.xbounds={}
        self.xstep={}
        self.xsort={}
        self.selweight={}       
        self.archive={}
        self.temp=[initemp*decrease**i for i in range(ntemp)]
        self.weight=[]
        self.niter=100
        self.archivefile="archive.json"
        self.archivesize=100
        self.maxarchivereject=1000
        self.alpha=0.0
        
    def settemp(self,initemp=1.0,decrease=0.9,ntemp=10,niter=100):
        '''
        settemp -> sets the fictitious temperature for Simulated Annealing.
        
        Parameters
        ----------
        initemp : double, optional
            Initial temperature for the Simulated Annealing algorithm. Default 
            value is 1.0.
        decrease : double, optional
            Decrease factor of the temperature during Simulated Annealing. It
            determines how fast the quench will occur. Default value is 0.9.
        ntemp : integer, optional
            Number of temperatures to be considered in Simulated Annealing.
            Default is 10.
        niter : integer, optional
            Number of iterations per temperature. Default is 100.
            
        Returns
        -------
        None.
        '''       
        print("Setting temperature...")
                
        if not isinstance(decrease,float) or decrease<=0.0 or decrease>=1.0:
            raise MOSAError("Decrease factor must be a float number greater than zero and less than one!")
                        
            return
        
        if not isinstance(ntemp,int) or ntemp<=0:
            raise MOSAError("Number of annealing temperatures must be an integer greater than zero!")
                        
            return
        
        if isinstance(initemp,(float,int)) and initemp>0.0:
            self.temp=[initemp*decrease**i for i in range(ntemp)]
        else:
            raise MOSAError("Initial temperature must be a float number greater than zero!")
                        
            return
        
        if isinstance(niter,int) and niter>0:
            self.niter=niter
        else:
            raise MOSAError("Number of iterations must be an integer greater than zero!")
            
            return
        
        print("Done!")
        
    def setarchiveconfig(self,archivesize=100,archivefile="archive.json",
                         maxarchivereject=1000):
        '''
        setarchiveconfig -> sets configurations of the archive.
        
        Parameters
        ----------
        archivesize : integer, optional
            Maximum number of solutions in the archive. Default value is 100.
        archivefile : string, optional
            Text file where the archive should be saved to. Default value is
            'archive.json'.
        maxarchivereject : integer, optional
            Maximum number of consecutive rejections of insertion of a solution 
            in the archive. Once reached, the optimization process finishes.
            Default value is 1000.
        
        Returns
        -------
        None. 
        '''                        
        print("Setting archive configurations...")
                
        if isinstance(archivesize,int) and archivesize>0:
            self.archivesize=archivesize
        else:
            raise MOSAError("The archive size must be an integer greater than zero!")
                        
            return
        
        if isinstance(archivefile,str) and len(archivefile)>0:
            self.archivefile=archivefile
        else:
            raise MOSAError("A filename must be provided!")
                        
            return            
        
        if isinstance(maxarchivereject,int) and maxarchivereject>0:
            self.maxarchivereject=maxarchivereject
        else:
            raise MOSAError("Maximum archive rejections must be an integer greater than zero!")
                        
            return
        
        print("Done!")
        
    def setxconfig(self,xnel={},xdistinct={},xstep={},exchangeprob={},
                   xbounds={},xsort={},selweight={}):
        '''
        setxconfig -> sets configurations of the solution.
        
        Parameters
        ----------
        xnel : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the number of elements for that key in
            the solution set. Default value is an empty dictionary.
        xdistinct : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies if the insertion of an element in that 
            key in the solution set requires the corresponding element to be 
            removed from the population. Default value is an empty dictionary.
        xstep : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution and specifies the maximum number of steps, to the left or
            to the right, that the algorithm can take when randomly selecting 
            an element in the corresponding key in the population to insert in 
            the solution. Default is an empty dictionary.
        exchangeprob : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the probability that elements will be
            exchanged between that key in the solution set and the population.
            If less than 1, it implies that there is a probability that elements
            will be added/removed from that key in the solution set. Default 
            value is an empty dictionary.
        xbounds : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the lower and upper values of solution
            elements. Default value is an empty dictionary.
        xsort : dictionary
            A Python dictionary where each key corresponds to a key in the
            solution set and specifies if the list in that key must be sorted
            in ascending order.
        selweight : dictionary
            A Python dictionary where each element corresponds to a key in the
            solution set and specifies the selection weight of this key in an
            MC iteration. Default value is an empty dictionary.
            
        Returns
        -------
        None.        
        '''
        print("Setting solution configurations...")

        if isinstance(xnel,dict):
            for key,val in xnel.items():
                if isinstance(val,int) and val>0: 
                    self.xnel[key]=val
                else:
                    raise MOSAError("xnel['%s'] must be an integer greater than zero!" 
                                    % key)
        else:
            raise MOSAError("'xnel' must be a dictionary!")
        
        if isinstance(xdistinct,dict):
            for key,val in xdistinct.items():
                if isinstance(val,bool):
                    self.xdistinct[key]=val
                else:
                    raise MOSAError("xdistinct['%s'] must be boolean!" % key)
        else:
            raise MOSAError("'xdistinct' must be a dictionary!")
            
        if isinstance(xstep,dict):   
            for key,val in xstep.items():
                if isinstance(val,(int,float)):
                    self.xstep[key]=val
                else:
                    raise MOSAError("xstep['%s'] must be either an integer or a float" 
                                    % key)
        else:
            raise MOSAError("'xstep' must be a dictionary!")
            
        if isinstance(exchangeprob,dict):
            for key,val in exchangeprob.items():
                if isinstance(val,float) and val>=0.0 and val<=1.0:
                    self.exchangeprob[key]=val
                else:
                    raise MOSAError("exchangeprob['%s'] must be a float number in the [0,1] interval!!" 
                                    % key)
        else:
            raise MOSAError("'exchangeprob' must be a dictionary!")
        
        if isinstance(xbounds,dict):
            for key,val in xbounds.items():
                if isinstance(val,list) and len(val)==2 and val[1]>val[0]:
                    self.xbounds[key]=val.copy()
                else:
                    raise MOSAError("xbounds['%s'][1] must be greater than xbounds['%s'][0]" 
                                     % key)
        else:
            raise  MOSAError("'xbounds' must be a dictionary!")
        
        if isinstance(xsort,dict):
            for key,val in xsort.items():
                if isinstance(val,bool):
                    self.xsort[key]=val
                else:
                    raise MOSAError("xsort['%s'] must be boolean!" % key)
        else:
            raise MOSAError("'xsort' must be a dictionary!")
         
        if isinstance(selweight,dict):
            for key,val in selweight.items():
                if isinstance(val,float):
                    self.selweight[key]=val
                else:
                    raise MOSAError("selweight[%s] must be a float number!" 
                                    % key)
        else:
            raise MOSAError("'selweight' must be a float number!")
            
        print("Done!")

    def setweight(self,weight):
        '''
        setweight -> sets the weights of the objectives to be minimized.
            
        Parameters
        ----------
        weight : list
            A Python list containing weights for the objectives, one per 
            objective.
        
        Returns
        -------
        None.        
        '''       
        print("Setting weights...")
                
        if isinstance(weight,list):
            self.weight=weight.copy()
        else:
            raise MOSAError("The array of weights must be a list!")
                        
            return
                
        print("Done!")
        
    def setpopulation(self,population):
        '''
        setpopulation -> sets the population.
            
        Parameters
        ----------
        population : dictionary
            A Python dictionary, each key of which contains the data that 
            can be used to achieve an optimized solution for the problem.
        
        Returns
        -------
        None.        
        '''               
        print("Setting population...")
                
        if isinstance(population,dict) and bool(population):
            self.population=deepcopy(population)
        else:
            raise MOSAError("Population must be a non-empty dictionary!")
                        
            return
        
        print("Done!")
        
    def setalpha(self,alpha=0.0):
        '''
        setalpha -> defines the alpha parameter used in MOSA.

        Parameters
        ----------
        alpha : float, optional
            Value of the alpha parameter. The default is 0.0.

        Returns
        -------
        None.
        '''
        print("Setting alpha...")        
        
        if isinstance(alpha,float) and alpha>=0.0 and alpha<=1.0:
            self.alpha=alpha
        else:            
            raise MOSAError("Alpha must be a float number between zero and one!")
            
            return
            
        print("Done!")

    def evolve(self,func):
        '''
        evolve -> performs the Multi-Objective Simulated Annealing (MOSA) 
            algorithm.
        
        Parameters
        ----------
        func : Python object
            A Python function that returns the value of the objective(s).
            
        Returns
        -------
        None.
        '''
        print("--- BEGIN: Evolving a solution ---\n")
          
        narchivereject=0
        fcurr=[]
        ftmp=[]
        weight=[]
        dstep={}
        lstep={}
        population={}
        poptmp={}
        xcurr={}
        xtmp={}
        xstep={}
        xbounds={}
        exchangeprob={}
        xdistinct={}
        xnel={}
        xsort={}
        totlength=0.0
        sellength={}        
        keys=[]
        from_checkpoint=False
        pmax=0.0
        
        self.__getarchive__()
        
        xcurr,fcurr,population=self.__getcheckpoint__()
        
        if bool(population) and bool(xcurr) and len(fcurr)>0:
            if set(population.keys())==set(xcurr.keys()):               
                from_checkpoint=True
        else:
            xcurr={}
            fcurr=[]
            population=deepcopy(self.population)
            
        keys=list(population.keys())
        
        print("------")        
        print("Population/solution keys:")
        
        for key in keys:
            print("    ['%s']:" % key)
            
            if len(population[key])==0:
                print("        Number of elements in the population: 0")
                print("        (Continuous sampling space)")
                
                if key in self.xbounds and len(self.xbounds[key])==2:
                    xbounds[key]=self.xbounds[key].copy()
                    
                    if xbounds[key][1]<=xbounds[key][0]:
                        xbounds[key]=[-1.0,1.0]
                else:
                    xbounds[key]=[-1.0,1.0]
                    
                print("        Boundaries: [%.4f,%.4f]" % 
                      (xbounds[key][0],xbounds[key][1]))
            else:                
                print("        Number of elements in the population: %d" 
                      % (len(population[key])))
                print("        (Discrete sampling space)")
                
                population[key].sort()
                
                if key in self.xdistinct:
                    xdistinct[key]=bool(self.xdistinct[key])
                else:
                    xdistinct[key]=False
                
                print("        Remove from the population the element selected in an MC iteration: %s"
                      % xdistinct[key])
                
            if key in self.selweight:
                totlength+=self.selweight[key]
                
                print("        Selection weight of this key: %f" 
                      % self.selweight[key])
            else:
                totlength+=1.0
            
                print("        Selection weight of this key: %f" % 1.0)  
            
            sellength[key]=totlength
                
            if key in self.exchangeprob and self.exchangeprob[key]>=0.0 \
                and self.exchangeprob[key]<=1.0:
                exchangeprob[key]=float(self.exchangeprob[key])
            else:
                exchangeprob[key]=1.0
                
            print("        Probability of element exchange between population and solution: %f" 
                  % (exchangeprob[key]*100.0))
            print("        Probability of element insertion to/deletion from solution: %f" 
                  % ((1.0-exchangeprob[key])*100.0))
                            
            if key in self.xsort:
                xsort[key]=bool(self.xsort[key])
            else:
                xsort[key]=False
                
            print("        Solution sorted after change: %s" 
                  % xsort[key])
                
            if key in self.xstep:
                if len(population[key])==0:
                    xstep[key]=float(self.xstep[key])
                    
                    if xstep[key]<=0.0:
                        xstep[key]=0.1
                        
                    print("        Maximum step size to choose a new value in the solution: %f" % xstep[key])
                else:
                    xstep[key]=int(self.xstep[key])
                    
                    if xstep[key]>len(population[key])/2 or xstep[key]<0:
                        xstep[key]=0
                    elif xstep[key]>0 and xstep[key]<=4:
                        xstep[key]=5
                        
                    if xstep[key]>=5:
                        dstep[key]=list(range(-xstep[key],xstep[key]+1))
                    
                    if xstep[key]==0:
                        print("        Element chosen at random in the population at every MC iteration")
                    else:
                        print("        Maximum step size to select an element in the population: %d" % xstep[key])
            else:
                if len(population[key])==0:
                    xstep[key]=0.1
                    
                    print("        Maximum step size to choose a new value in the solution: %f"
                          % 0.1)
                else:
                    xstep[key]=0
                    
                    print("        Element chosen at random in the population at every MC iteration")
                    
        print("------")
                
        if from_checkpoint:
            print("Initial solution from the checkpoint file...")
        else:
            print("Initializing with a random solution from scratch...")
            
            for key in keys:
                xcurr[key]=[]
                
                if key in self.xnel and self.xnel[key]>0:
                    xnel[key]=self.xnel[key]
                else:
                    xnel[key]=1
                
                for j in range(xnel[key]):
                    if len(population[key])>0:
                        m=choice(len(population[key]))
                        xcurr[key].append(population[key][m])
                    
                        if xdistinct[key]:
                            population[key].pop(m)
                    else:                            
                        xcurr[key].append(uniform(xbounds[key][0],
                                                            xbounds[key][1]))
                        
                if xsort[key]:
                    xcurr[key].sort()
            
            if callable(func):
                fcurr=func(xcurr)             
                updated=self.__updatearchive__(xcurr,fcurr)
            else:
                raise MOSAError("A Python function must be provided!")
                    
                return
            
        print("Done!")
        print("------")
            
        if len(fcurr)==len(self.weight):
            weight=self.weight.copy()
        else:
            weight=[1.0 for k in range(len(fcurr))]
            
        for key in keys:
            if len(population[key])>0:
                lstep[key]=choice(len(population[key]))
                    
        for temp in self.temp:
            print("TEMPERATURE: %.6f" % temp)
            
            nupdated=0
            naccept=0
            
            for j in range(self.niter):
                r=uniform(0.0,totlength)
                
                for key in keys:
                    if r<sellength[key]:
                        break
                
                xtmp=deepcopy(xcurr)
                poptmp=deepcopy(population)
                old=choice(len(xtmp[key]))
                
                if len(poptmp[key])>0:                   
                    if xstep[key]==0:
                        new=choice(len(poptmp[key]))
                    else:                       
                        new=lstep[key]+choice(dstep[key])
                        
                        if new>=len(poptmp[key]):
                            new-=len(poptmp[key])
                        elif new<0:
                            new+=len(poptmp[key])
                            
                if uniform(0.0,1.0)<exchangeprob[key]:
                    if len(poptmp[key])>0:
                        popel=poptmp[key][new]
                        xel=xtmp[key][old]
                        xtmp[key][old]=popel
                                
                        if xdistinct[key]:
                            poptmp[key][new]=xel
                            poptmp[key].sort()
                    else:
                        xtmp[key][old]+=uniform(-xstep[key],xstep[key])
                        
                        if xtmp[key][old]>xbounds[key][1]:
                            xtmp[key][old]-=(xbounds[key][1]-xbounds[key][0])
                        elif xtmp[key][old]<xbounds[key][0]:
                            xtmp[key][old]+=(xbounds[key][1]-xbounds[key][0])
                            
                    if xsort[key]:
                        xtmp[key].sort()
                else:
                    if len(xtmp[key])==1:
                        r=0.0
                    elif len(poptmp[key])==1 and xdistinct[key]:
                        r=1.0
                    else:
                        r=uniform(0.0,1.0)
                    
                    if r<0.5:
                        if len(poptmp[key])>0:
                            xtmp[key].append(poptmp[key][new])
                        
                            if xdistinct[key]:
                                poptmp[key].pop(new)
                        else:
                            xtmp[key].append(uniform(xbounds[key][0],
                                                     xbounds[key][1]))
                            
                        if xsort[key]:
                            xtmp[key].sort()
                    else:
                        if len(poptmp[key])>0 and xdistinct[key]:
                            poptmp[key].append(xtmp[key][old])
                            poptmp[key].sort()
                            
                        xtmp[key].pop(old)
                                        
                gamma=1.0
                ftmp=func(xtmp)
                
                for k in range(len(ftmp)):
                    if ftmp[k]<fcurr[k]:                        
                        pmax=p=1.0                        
                    else:
                        p=exp(-(ftmp[k]-fcurr[k])/(temp*weight[k]))
                        
                        if pmax<p:
                            pmax=p
                            
                    gamma*=p
                    
                gamma=(1.0-self.alpha)*gamma+self.alpha*pmax
                
                if gamma==1.0 or uniform(0.0,1.0)<gamma:
                    if len(poptmp[key])>0:                        
                        if new<len(poptmp[key]):
                            lstep[key]=new
                        else:
                            lstep[key]=0
                                          
                    fcurr=ftmp.copy()
                    xcurr=deepcopy(xtmp)
                    population=deepcopy(poptmp)                    
                    naccept+=1                    
                    updated=self.__updatearchive__(xcurr,fcurr)
                    nupdated+=updated
                    
                    self.__savecheckpoint__(xcurr,fcurr,population)
                    
                    if updated==0:
                        narchivereject+=1
                        
                        if narchivereject==self.maxarchivereject:
                            print("    Insertion in the archive rejected too many times!")
                            print("    Quiting at iteration %d..." % j)
                            print("------")
                            print("\n--- THE END ---")
                            
                            self.__savearchive__()
                            
                            return
                    else:
                        narchivereject=0
                           
            if naccept>0:
                print("    Number of accepted moves: %d." % naccept)                   
                print("    Fraction of accepted moves: %.6f." % 
                      (naccept/self.niter))
            
                if nupdated>0:
                    print("    Number of archive updates: %d." % nupdated)
                    print("    Fraction of archive updates in accepted moves: %.6f." % 
                          (nupdated/naccept))
                    
                    self.__savearchive__()
                else:
                    print("    No archive update.")
            else:
                print("    No move accepted.")
                
            print("------")
            
        print("\n--- THE END ---")
        
    def purgedominated(self,xset={}):
        '''
        purgedominated -> returns a subset of the full or reduced archive that
        contains only non-dominated solutions.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are non-dominated.
        '''       
        tmpdict={}
        tmpdict["Solution"]=[]
        tmpdict["Values"]=[]
        
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return {}
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return {}
                
        included=[True for i in range(len(xset["Values"]))]
                        
        for i in range(len(xset["Values"])-1):
            if not included[i]:
                continue
            
            for j in range(i+1,len(xset["Values"])):
                if not included[j]:
                    continue
                
                nl=ng=0
                
                for k in range(len(xset["Values"][i])):
                    if xset["Values"][i][k]<xset["Values"][j][k]:
                        nl+=1
                    elif xset["Values"][i][k]>xset["Values"][j][k]:
                        ng+=1
                        
                if nl>0 and ng==0:
                    included[j]=False
                elif ng>0 and nl==0:
                    included[i]=False
                    
                    break
                    
        for i in range(len(xset["Values"])):
            if included[i]:
                tmpdict["Solution"].append(deepcopy(xset["Solution"][i]))
                tmpdict["Values"].append(xset["Values"][i].copy())
                
        return tmpdict
        
    def trimx(self,xset={},thresholds=[]):
        '''
        trimx -> extracts from the archive the solutions whose values of the
        objective functions are less than the given threshold values.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.
        thresholds : list, optional
            Maximum values of the objective funcions required for a solution
            to be selected. The default is an empty list.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are in agreement with the thresholds.
        '''
        tmpdict={}
        tmpdict["Solution"]=[]
        tmpdict["Values"]=[]
        
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return {}
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return {}
                
        indexlist=list(range(len(xset["Values"])))
        
        if len(thresholds)==len(xset["Values"][0]):
            for i in indexlist:            
                for j in range(len(xset["Values"][i])):
                    if xset["Values"][i][j]<=thresholds[j]:
                        included=True
                    else:
                        included=False
                    
                        break
                
                if included:
                    tmpdict["Solution"].append(deepcopy
                                               (xset["Solution"][i]))
                    tmpdict["Values"].append(xset["Values"][i].copy())
        else:
            raise MOSAError("The threshold list cannot be empty!")
            
            return {}
                
        return tmpdict
        
    def reducex(self,xset={},index=0,nel=5):
        '''
        reducex -> reduces and sorts the archive according to the value of
        an objective function.        

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.
        index : integer, optional
            Index of the objective function value that will be used when 
            comparing solutions that will be sorted and introduced in the
            reduced solution archive. The default is 0.
        nel : integer, optional
            Number of solutions stored in the reduced archive. The default is 5.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the reduced solution archive.
        '''
        tmpdict={}
        tmpdict["Solution"]=[]
        tmpdict["Values"]=[]
        
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return {}
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return {}
            
        if nel>len(xset["Values"]):
            nel=len(xset["Values"])
            
        indexlist=list(range(len(xset["Values"])))
        
        for i in range(nel):
            k=0
            
            for j in indexlist:
               if k==0:
                   toadd=j
                   bestval=xset["Values"][j][index]
                   k+=1
               else:
                   if xset["Values"][j][index]<bestval:
                       toadd=j
                       bestval=xset["Values"][j][index]
                       
            tmpdict["Solution"].append(deepcopy(xset["Solution"][toadd]))
            tmpdict["Values"].append(xset["Values"][toadd].copy())
            indexlist.remove(toadd)

        return tmpdict
    
    def printx(self,xset={}):
        '''
        printx -> prints the solutions in the archive (complete or reduced) in 
        a more human readable format.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary, 
            meaning the full solution archive.
        
        Returns
        -------
        None.
        '''
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return
        
        print("===")
        print("Solutions:")
        
        for i in range(len(xset["Solution"])):
            print("%d) %s" % (i+1,xset["Solution"][i]))

        print("Values:")
        
        for i in range(len(xset["Values"])):
            print("%d) %s" % (i+1,xset["Values"][i]))
        
    def plotfront(self,xset={},index1=0,index2=1):
        '''
        plotfront -> plots 2D scatter plots of selected pairs of objective 
        functions.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary, 
            meaning the full solution archive.
        index1 : integer, optional
            Index of the objective function the value of which will be 
            displayed along x-axis. The default is 0.
        index2 : integer, optional
            Index of the objective function the value of which will be 
            displayed along y-axis. The default is 1.

        Returns
        -------
        None.
        '''
        try:
            import matplotlib.pyplot as plt
        except:
            raise MOSAError("Matplotlib is not available in your system!")
            
            return
        
        f=[[],[]]
        
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return
        
        if index1>=0 and index1<len(xset["Values"][0]) and index2>=0 and \
            index2<len(xset["Values"][0]):
            for i in range(len(xset["Values"])):
                f[0].append(xset["Values"][i][index1])
                f[1].append(xset["Values"][i][index2])
                
            plt.xlabel("f%d" % index1)
            plt.ylabel("f%d" % index2)
            plt.grid()
            plt.scatter(f[0],f[1])
            plt.show()
        else:
            raise MOSAError("Index out of range!")
            
            return
        
    def dumpfront(self,xset={},filename="dump.csv"):
        '''
        dumpfront -> dumps the solutions in the archive (complete or reduced) 
        into a CSV file for further processing.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary, 
            meaning the full solution archive.
        filename : str, optional
            Name of the CSV file. The default is "dump.csv".

        Returns
        -------
        None.
        '''
        if not bool(xset):
            xset=self.archive
        else:
            if not ("Solution" in xset and "Values" in xset):
                raise MOSAError("'Solution' and 'Values' must be present in the dictionary!")
                
                return
            else:
                if not (isinstance(xset["Solution"],list) and 
                        isinstance(xset["Values"],list)):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
                    
                    return
        
        with open(filename,"w") as f:
            header="Index"
            nobj=len(xset["Values"][0])
                     
            for i in range(nobj):
                header+=",f%d" % i
                
            f.write(header+"\n")
            
            for i in range(len(xset["Values"])):
                line="%d" % i
            
                for fobj in xset["Values"][i]:
                    line+=",%f" % fobj
            
                f.write(line+"\n")
            
    def __updatearchive__(self,x,f):
        '''
        __updatearchive__ -> checks if the solution given as argument is better 
            than solutions randomly (and sequentially) chosen from the archive. 
            If so, the archive is updated, this solution is appended and a 
            worse solution is removed.
            
        Parameters
        ----------
        x : dictionary
            A Python dictionary containing the solution.
        f : list
            A Python list containing the values of the objectives associated
            with the solution.
            
        Returns
        -------
        updated : integer
            1, if the archive is updated, or 0, if not.
        '''
        updated=0
        indexlist=list(range(len(self.archive["Values"])))
        
        for i in indexlist:
            if f==self.archive["Values"][i]:
                return updated
        
        if len(self.archive["Solution"])==0:            
            updated=1
        else:
            shuffle(indexlist)
            
            for i in indexlist:
                nl=ng=0
                
                for j in range(len(self.archive["Values"][i])):
                    if f[j]<self.archive["Values"][i][j]:
                        nl+=1
                    elif f[j]>self.archive["Values"][i][j]:
                        ng+=1
                        
                if len(self.archive["Solution"])<self.archivesize:
                    if nl>0:
                        updated=1
                        
                        if ng==0:
                            self.archive["Solution"].pop(i)
                            self.archive["Values"].pop(i)
                            
                            break
                    else:
                        updated=0
                        
                        break
                else:
                    if nl>0 and ng==0:
                        self.archive["Solution"].pop(i)
                        self.archive["Values"].pop(i)
                        
                        updated=1
                    
                        break
                        
        if updated==1:
            self.archive["Solution"].append(x)
            self.archive["Values"].append(f)
        
        return updated
    
    def __getarchive__(self):
        '''
        __getarchive__ -> initializes the archive.
                
        Returns
        -------
        None.
        '''
        print("Initializing archive...")
                
        try:
            self.archive=json.load(open(self.archivefile,"r"))
            
            if "Solution" in self.archive and "Values" in self.archive:
                print("Archive loaded from "+self.archivefile+"!")
            else:
                self.archive["Solution"]=[]
                self.archive["Values"]=[]
                
                print("WARNING: Starting with an empty archive because the archive loaded from "+self.archivefile+" has a wrong format!")
        except:
            self.archive["Solution"]=[]
            self.archive["Values"]=[]
            
            print("Empty archive!")
        
        print("Done!")
        
    def __savearchive__(self):
        '''
        __savearchive__ -> saves the archive as JSON into a text file.
        
        Returns
        -------
        None.
        '''
        json.dump(self.archive,open(self.archivefile,"w"),indent=4)
        
    def __getcheckpoint__(self):
        '''
        __getcheckpoint__ -> initializes with a solution from a previous run.
                
        Returns
        -------
        x : dictionary
            A Python dictionary containing a solution.
        f : list
            A Python list containing the values of the objective functions
            associated with the solution.
        population : dictionary
            A Python dictionary containing the population compatible with the
            solution.
        '''
        tmpdict={}
        x={}
        f=[]
        population={}
        
        print("Looking for a solution in the checkpoint file...")
           
        try:            
            tmpdict=json.load(open("checkpoint.json","r"))
            
            if "Solution" in tmpdict and "Values" in tmpdict and \
                "Population" in tmpdict:
                x=deepcopy(tmpdict["Solution"])
                f=tmpdict["Values"].copy()
                population=deepcopy(tmpdict["Population"])
        except:          
            print("No checkpoint file!")
        
        print("Done!")
        
        return x,f,population
    
    def __savecheckpoint__(self,x,f,population):
        '''
        __savecheckpoint__ -> saves the solution passed as argument as JSON 
        into a text file.

        Parameters
        ----------        
        x : dictionary
            A Python dictionary containing the solution.
        f : list
            A Python list containing the values of the objectives associated
            with the solution.
        population : dictionary
            A Python dictionary containing the population compatilbe with the 
            solution.
        
        Returns
        -------
        None.
        '''
        tmpdict={"Solution":x,"Values":f,"Population":population}
        
        json.dump(tmpdict,open("checkpoint.json","w"),indent=4)
    
class MOSAError(Exception):
    def __init__(self,message=""):
        '''
        __init__ -> class constructor.

        Parameters
        ----------
        message : string, optional
            Error message to be displayed. The default is "".

        Returns
        -------
        None.
        '''
        self.message=message
      
    def __str__(self):
        '''
        __str__ -> returns the error message.

        Returns
        -------
        message
            Error message to be displayed.

        '''
        return self.message
