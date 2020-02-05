# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:50:07 2020

@author: bruger

Handeling of models with equiblibium condition 
for instance supply = demand 

we have to handle 
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import rc
import sys
import re
import scipy as sp
import os
from collections import namedtuple
from sympy import sympify,Symbol




import modelclass as mc
from modelclass import ttimer
import modelsandbox as ms
from modelsandbox import newmodel, new_newton_diff
import modelmanipulation as mp 
from modelmanipulation import split_frml,udtryk_parse
import modelpattern as pt 
import modeldiff as md 


def findindex(ind00):
    ''' 
     - an equation looks like this
     - <frmlname> [index] lhs = rhs 
    
    this function find frmlname and index variables on the left hand side. meaning variables braced by {} '''
    ind0 = ind00.strip()
    if ind0.startswith('<'):
        frmlname = re.findall('\<.*?\>',ind0)[0]
        ind = ind0[ind0.index('>')+1:].strip()
    else:
        frmlname='<>'
        ind=ind0.strip()
        
    if ind.startswith('['):
        allindex = re.findall('\[.*?\]',ind0)[0]
        index = allindex[1:-1].split(',')
        rest = ind[ind.index(']')+1:]
    else:
        index = []
        rest = ind
    return frmlname,index,rest

def un_normalize(frml) :
    '''This function makes sure that all formulas are unnormalized.
    if the formula is already decorated with <endo=name> this is keept 
    else the lhs_varriable is used in <endo=> 
    ''' 
    frml_name,frml_index,frml_rest = findindex(frml.upper())
    this_endo = pt.kw_frml_name(frml_name.upper(), 'ENDO')
    lhs,rhs  = frml_rest.split('=')
    if this_endo: 
        lhs_var = this_endo.strip()
        frml_name_out = frml_name
    else:
        lhs_var = lhs.strip()
        frml_name_out = f'<endo={lhs_var}>' if frml_name == '<>' else f'{frml_name[:-1]},endo={lhs_var}>'
    print(this_endo)
    new_rest = f'{lhs_var}___res = ( {rhs.strip()} ) - ( {lhs.strip()} )'
    return f'{frml_name_out} {frml_index if len(frml_index) else ""} {new_rest}'
        

if __name__ == '__main__':
    #%% specify datataframe

    demandparam = 1,-0.4,0.0
    supplyparam = 0.0,0.5,0.
    col =[c.upper() for c in  ['demand_ofset','demand_slope','demand_second',
           'supply_ofset','supply_slope','supply_second','price']]
    
    grunddf = pd.DataFrame(index=list(range(3)),columns=col)
    
    grunddf.loc[:,col] = demandparam+supplyparam+(4.,)
    grunddf.loc[:,'DEMAND_OFSET']=[1.0,0.9,0.8]
    rdm = '''\
                 demand = (demand_ofset+ demand_slope*price+ demand_second * price**2)
                 supply = supply_ofset+ supply_slope*price+ supply_second * price**2
    <endo=price> supply = demand 
    
    '''.upper()
    rdm2 = '''\
    <endo=price> supply_ofset+ supply_slope*price+ supply_second * price**2 = (demand_ofset+ demand_slope*price+ demand_second * price**2) 
    '''
    os.environ['PYTHONBREAKPOINT'] = '0'

    edm  = '\n'.join(un_normalize(f) for f in rdm.split('\n') if len(f.strip()))
    fdm  = mp.explode(edm)
    mdm = ms.newmodel(fdm)
    #%%
    grunddf = mc.insertModelVar(grunddf,mdm).astype('float')
    sim1 = mdm.newton1per(grunddf,antal=30,silent=0,nonlin=0,newtonalfa=0.4,newtondamp=4)
    ssim1 = mdm.newtonstack(grunddf,antal=30,silent=0,nonlin=0,newtonalfa=1,newtondamp=4)
    #%%
   # print(newton.diff_model.equations)
    if 0:
        mat1pjac = newton.get_jacdf()
        diffrml = newton.diff_model.equations
        difres_before = newton.difres_before
        sp.sparse.csc_matrix([1,2],[(1,2),(4,4)],shape=(30,30))
    
    
  