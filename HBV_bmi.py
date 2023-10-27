from empty_bmi import EmptyBmi
from typing import Tuple
import numpy as np
import pandas as pd
import json


class HBV(EmptyBmi):
    """HBV model wrapped in a BMI interface."""

    def initialize(self, config_file: str) -> None:
        # open json files containing data
        f = open(config_file)
        self.config: dict[str, Any] = json.load(f)
        f.close()

        # load data frame with pandas
        df_in = pd.read_csv(self.config['forcing_file'], delimiter="	", names=["year","month","day","P","Q","EP"])
        df_in.index = df_in.apply(lambda x: pd.Timestamp(f'{int(x.year)}-{int(x.month)}-{int(x.day)}'),axis=1)
        df_in.drop(columns=["year","month","day"],inplace=True)
        self.df = df_in     

        # set up times
        self.end_timestep = len(self.df)+1
        self.current_timestep = 0
        self.dt = self.config['timestep']

        # define parameters 
        self.set_pars(np.array(self.config['parameters'].split(','),dtype=np.float64))
        
        # add memory vector for tlag & run weights function
        self.memory_vector_lag = np.zeros(self.T_lag)
        self.weights = self.Weigfun()

        # define storage & flow terms, flows 0, storages initialised 
        s_in = np.array(self.config['initial_storage'].split(','),dtype=np.float64)
        self.set_storage(s_in)
        # set other flows for initial step 
        self.Ei_dt     = 0       # interception evaporation
        self.Ea_dt     = 0       # actual evaportation
        self.Qs_dt     = 0       # slow flow 
        self.Qf_dt     = 0       # fast flow
        self.Q_tot_dt  = 0       # total flow
        self.Q_m       = 0       # Final model prediction
    
    def set_pars(self, par) -> None:
        # function to overwrite initial configureation of parameters, saves having to change the config file
        self.I_max  = par[0]                # maximum interception
        self.Ce     = par[1]                # Ea = Su / (sumax * Ce) * Ep
        self.Su_max = par[2]                # ''
        self.beta   = par[3]                # Cr = (su/sumax)**beta
        self.P_max  = par[4]                # Qus = Pmax * (Su/Sumax)
        self.T_lag  = self.set_tlag(par[5]) # used in triangular transfer function
        self.Kf     = par[6]                # Qf=kf*sf
        self.Ks     = par[7]                # Qs=Ks*

    def set_storage(self, stor) -> None:
        self.Si = stor[0] # Interception storage
        self.Su = stor[1] # Unsaturated Rootzone Storage
        self.Sf = stor[2] # Fastflow storage
        self.Ss = stor[3] # Groundwater storage       
        

    def update(self) -> None:
        """
        Function to run the updatre part of one timestep of the HBV model     
        parameters
        ----------
        par: array_like
            array/list containing 8 parameters: Imax,  Ce,  Sumax, beta,  Pmax,  T_lag,   Kf,   Ks (floats)
        df: pandas.core.frame.DataFrame
            DataFrame containing 'P', 'Q', 'EP' columns as forcing for the model.
        s_in: array_like
            array/list containing 4 storage terms which are input to the timestep: Si,  Su, Sf, Ss (floats)
        storage_terms: list of arrays
            list of arrays which store: Si, Su, Sf, Ss, Ei_dt, Ea_dt, Qs_dt_lst, Qf_dt_lst, Q_tot_dt
        step_n: int
            nth step which formard model takes: used to determin which Precipitaion & evaporation to use
    
        Returns
        ----------
        Obj, df: float, pandas.core.frame.DataFrame
            return of the objective function and corresponding dataframe
    
        """
        if self.current_timestep <= self.end_timestep:
            self.P_dt  = self.df.P.iloc[self.current_timestep] * self.dt
            self.Ep_dt = self.df.EP.iloc[self.current_timestep]  * self.dt
        
            # Interception Reservoir
            if self.P_dt > 0:
                # if there is rain, no evap
                self.Si    = self.Si + self.P_dt               # increase the storage
                self.Pe_dt = max((self.Si - self.I_max) / self.dt, 0)
                self.Si    = self.Si - self.Pe_dt 
                self.Ei_dt = 0                          # if rainfall, evaporation = 0 as too moist 
            else:
                # Evaporation only when there is no rainfall
                self.Pe_dt    = 0                      # nothing flows in so must be 0
                self.Ei_dt    = min(self.Ep_dt, self.Si / self.dt) # evaporation limited by storage
                self.Si    = self.Si - self.Ei_dt 
        
            # split flow into Unsaturated Reservoir and Fast flow
            if self.Pe_dt > 0:
                cr       = (self.Su / self.Su_max)**self.beta
                Qiu_dt   = (1 - cr ) * self.Pe_dt      # flux from Ir to Ur
                self.Su  = self.Su + Qiu_dt
                Quf_dt   = cr  * self.Pe_dt            # flux from Su to Sf
            else:
                Quf_dt = 0
        
            # Transpiration
            self.Ep_dt = max(0, self.Ep_dt - self.Ei_dt)        # Transpiration 
            self.Ea_dt = self.Ep_dt  * (self.Su / (self.Su_max * self.Ce))
            self.Ea_dt = min(self.Su, self.Ea_dt)            # limited by water in soil 
            self.Su    = self.Su - self.Ea_dt
        
            # Percolation
            Qus_dt = self.P_max * (self.Su / self.Su_max) * self.dt # Flux from Su to Ss
            self.Su  = self.Su - Qus_dt
        
            # Fast Reservoir
            self.Sf = self.Sf + Quf_dt
            self.Qf_dt = self.dt * self.Kf * self.Sf
            self.Sf = self.Sf - self.Qf_dt
        
            # Slow Reservoir
            self.Ss = self.Ss + Qus_dt
            self.Qs_dt = self.Ss * self.Ks * self.dt
            self.Ss = self.Ss - self.Qs_dt
    
            # total = fast + slow
            self.Q_tot_dt = self.Qs_dt + self.Qf_dt
            # add time lag to the process
            self.add_time_lag()
            # self.Q_m = self.Q_tot_dt
            
            # Advance the model time by one step
            self.current_timestep += 1

    def Weigfun(self): 
        nmax=int(np.ceil(self.T_lag))
        if nmax==1: 
            Weigths=float(1)
        else:
            Weigths=np.zeros(nmax)
            th=self.T_lag/2
            nh=int(np.floor(th))
            for i in range(0,nh): 
                Weigths[i]=(float(i+1)-0.5)/th         
            i=nh
    
            Weigths[i]=(1+(float(i+1)-1)/th)*(th-int(np.floor(th)))/2+(1+(self.T_lag-float(i+1))/th)*(int(np.floor(th))+1-th)/2
            for i in range(nh+1, int(np.floor(self.T_lag))):
                Weigths[i]=(self.T_lag-float(i+1)+.5)/th
    
            if self.T_lag>int(np.floor(self.T_lag)):
                Weigths[int(np.floor(self.T_lag))]=(self.T_lag-int(np.floor(self.T_lag)))**2/(2*th)
    
            Weigths=Weigths/sum(Weigths)
        
        return Weigths
    
    def add_time_lag(self) -> None:
        # with support for T_lag =0
        if len(self.memory_vector_lag) > 0:        
            # Distribute current Q_tot_dt to memory vector
            self.memory_vector_lag += self.weights*self.Q_tot_dt
            
            # Extract the latest Qm
            self.Q_m = self.memory_vector_lag[0]
                     
            # Make a forecast to the next time step
            self.memory_vector_lag = np.roll(self.memory_vector_lag,-1)  # This cycles the array [1,2,3,4] becomes [2,3,4,1]
            self.memory_vector_lag[-1] = 0                              # the next last entry becomes 0 (outside of convolution lag)

    def get_component_name(self) -> str:
        return "HBV"

    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        if(var_name == "Imax"):
            dest[:] = np.array(self.Imax)
            return dest
        elif(var_name == "Ce"):
            dest[:] = np.array(self.Ce)
            return dest
        elif(var_name == "Sumax"):
            dest[:] = np.array(self.Su_max)
            return dest
        elif(var_name == "Beta"):
            dest[:] = np.array(self.beta)
            return dest
        elif(var_name == "Pmax"):
            dest[:] = np.array(self.P_max)
            return dest
        elif(var_name == "Tlag"):
            dest[:] = np.array(int(self.T_lag))
            return dest
        elif(var_name == "Kf"):
            dest[:] = np.array(self.Kf)
            return dest    
        elif(var_name == "Ks"):
            dest[:] = np.array(self.ks)
            return dest
        elif(var_name == "Si"):
            dest[:] = np.array(self.Si)
            return dest
        elif(var_name == "Su"):
            dest[:] = np.array(self.Su)
            return dest
        elif(var_name == "Sf"):
            dest[:] = np.array(self.Sf)
            return dest
        elif(var_name == "Ss"):
            dest[:] = np.array(self.Ss)
            return dest
        elif(var_name == "Ei_dt"):
            dest[:] = np.array(self.Ei_dt)
            return dest
        elif(var_name == "Ea_dt"):
            dest[:] = np.array(self.Ei_dt)
            return dest
        elif(var_name == "Qs_dt"):
            dest[:] = np.array(self.Qs_dt)
            return dest
        elif(var_name == "Qf_dt"):
            dest[:] = np.array(self.Qf_dt)
            return dest
        elif(var_name == "Q_tot_dt"):
            dest[:] = np.array(self.Q_tot_dt)
            return dest
        elif(var_name == "Q_m"):
            dest[:] = np.array(self.Q_m)
            return dest
        elif(var_name == "storage_terms"):
            dest[:] = np.array([self.Si, self.Su, self.Sf, self.Ss])
            return dest
        elif(var_name == "parameters"):
            dest[:] = np.array([self.I_max, self.Ce, self.Su_max, self.beta, self.P_max, self.T_lag, self.Kf, self.Ks])
            return dest
        elif(var_name == "memory_vector_lag"):
            return self.memory_vector_lag 
        else:
            raise ValueError(f"Unknown variable {var_name}")

    def get_var_units(self, var_name: str) -> str:
        if(var_name == "Imax"):
            return "mm"
        elif(var_name == "Ce"):
            return "-"
        elif(var_name == "Sumax"):
            return "mm"
        elif(var_name == "Beta"):
            return "-"
        elif(var_name == "Pmax"):
            return "mm"
        elif(var_name == "Tlag"):
            return "d"
        elif(var_name == "Kf"):
            return "-"
        elif(var_name == "Ks"):
            return "-"
        elif(var_name == "Si"):
            return "mm"
        elif(var_name == "Su"):
            return "mm"
        elif(var_name == "Sf"):
            return "mm"
        elif(var_name == "Ss"):
            return "mm"
        elif(var_name == "Ei_dt"):
            return "mm/d"
        elif(var_name == "Ea_dt"):
            return "mm/d"
        elif(var_name == "Qs_dt"):
            return "mm/d"
        elif(var_name == "Qf_dt"):
            return "mm/d"
        elif(var_name == "Q_tot_dt"):
            return "mm/d"
        elif(var_name == "Q_m"):
            return "mm/d"
        else:
            raise ValueError(f"Unknown variable {var_name}")

    def set_value(self, var_name: str, src: np.ndarray) -> None:
        if(var_name == "Imax"):
            self.Imax = src[0]
        elif(var_name == "Ce"):
            self.Ce = src[0]
        elif(var_name == "Sumax"):
            self.Su_max = src[0]
        elif(var_name == "Beta"):
            self.beta = src[0]
        elif(var_name == "Pmax"):
            self.P_max = src[0]
        elif(var_name == "Tlag"):
            self.T_lag = self.set_tlag(src[0])
        elif(var_name == "Kf"):
            self.Kf = src[0]
        elif(var_name == "Ks"):
            self.Ks = src[0]
        elif(var_name == "Si"):
            self.Si = src[0]
        elif(var_name == "Su"):
            self.Su = src[0]
        elif(var_name == "Sf"):
            self.Sf = src[0]
        elif(var_name == "Ss"):
            self.Ss = src[0]
        elif(var_name == "Ei_dt"):
            self.Ei_dt = src[0]
        elif(var_name == "Ea_dt"):
            self.Ea_dt = src[0]
        elif(var_name == "Qs_dt"):
            self.Qs_dt = src[0]
        elif(var_name == "Qf_dt"):
            self.Qf_ft = src[0]
        elif(var_name == "Q_tot_dt"):
            self.Q_tot_dt = src[0]
        elif(var_name == "Q_m"):
            self.Qm = src[0]
        elif(var_name == "memory_vector_lag"):
            self.memory_vector_lag = src
        else:
            raise ValueError(f"Unknown variable {var_name}")

    def get_output_var_names(self) -> Tuple[str]:
        return ("Imax","Ce","Sumax","Beta","Pmax","Tlag","Kf","Si","Su","Sf","Ss","Ei_dt","Ea_dt","Qs_dt","Qf_dt","Q_tot_dt","Q_m")

    # The BMI has to have some time-related functionality:
    def get_start_time(self) -> float:
        """Return end time in seconds since 1 january 1970."""
        return self.get_unixtime(self.df.index[0]) # type: ignore

    def get_end_time(self) -> float:
        """Return end time in seconds since 1 january 1970."""
        return get_unixtime(self.df.index[-1]) # type: ignore

    def get_current_time(self) -> float:
        """Return current time in seconds since 1 january 1970."""
        # we get the timestep from the data, but the stopping condition requires it to go one beyond. 
        if self.current_timestep < len(self.df):
            return get_unixtime(self.df.index[self.current_timestep]) # type: ignore
        else:
            return get_unixtime(self.df.index[-1] + pd.Timedelta(days=1))
    # return get_unixtime(self.df.loc[pd.Timestamp(model.get_current_time(), unit="s")]) # type: ignore
    
    def set_tlag(self, T_lag) -> int:
        "Ensures T_lag is an integer"
        if type(T_lag) != int:
            return int(T_lag)
        else:
            return T_lag

def get_unixtime(ts: pd.Timestamp) -> float:
    """Get unix timestamp (seconds since 1 january 1970) from a np.datetime64."""
    return  np.datetime64(pd.to_datetime(ts)).astype("datetime64[s]").astype("int")