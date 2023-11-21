#!/usr/bin/env python3.10


import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from pystyle.annotate import placeText
m_jpsi=3.1
p_jpsi=9
m_e=0.511*10**-3


plt.subplots(layout="constrained",figsize=(6,5))

for theta_em in [3,4,5]:
    # def equations(params):
    #     p_em,p_ep,theta_ep=params
    #     p_em=abs(p_em)
    #     p_ep = abs(p_ep)
    #     return (m_jpsi**2 - 2 * m_e**2- 2* p_em*p_ep*np.cos((theta_em-theta_ep)*np.pi/180),
    #             p_jpsi-p_em*np.cos(theta_em*np.pi/180)-p_ep*np.cos(theta_ep*np.pi/180),
    #             p_em * np.sin(theta_em * np.pi / 180) + p_ep * np.sin(theta_ep * np.pi / 180)
    #             )

    def equations(params):
        p_em,p_ep,theta_ep=params
        p_em=abs(p_em)
        p_ep = abs(p_ep)
        return (np.sqrt(m_jpsi**2 + p_jpsi**2)- np.sqrt(m_e**2 + p_em**2)-np.sqrt(m_e**2 + p_ep**2),
                p_jpsi-p_em*np.cos(theta_em*np.pi/180)-p_ep*np.cos(theta_ep*np.pi/180),
                p_em * np.sin(theta_em * np.pi / 180) + p_ep * np.sin(theta_ep * np.pi / 180)
                )

    p_em,p_ep,theta_ep =  fsolve(equations, (9,1,-40))

    print(p_em,p_ep,theta_ep)

    print(equations((p_em,p_ep,theta_ep)))
    def invMass_theta(dthedas):
        m_jspi_recon=np.zeros(len(dthedas))
        for i,dtheta in enumerate(dthedas):
            theta_em_deflect = theta_em + dtheta
            p4_em=[np.sqrt(m_e**2+p_em**2),0,p_em*np.sin(theta_em_deflect*np.pi/180),p_em*np.cos(theta_em_deflect*np.pi/180)]
            p4_ep=[np.sqrt(m_e**2+p_ep**2),0,p_ep*np.sin(theta_ep*np.pi/180),p_ep*np.cos(theta_ep*np.pi/180)]
            p4_jspi_recon=np.array(p4_em)+np.array(p4_ep)
            m_jspi_recon[i]=np.sqrt(p4_jspi_recon[0]**2-p4_jspi_recon[1]**2-p4_jspi_recon[2]**2-p4_jspi_recon[3]**2)
            # m_jspi_recon[i]=np.sqrt(2 * m_e ** 2 + 2 * p_em * p_ep * np.cos((theta_em + dtheta - theta_ep) * np.pi / 180))
        return m_jspi_recon



    def invMass_p_em(dp_ems):
        m_jspi_recon=np.zeros(len(dp_ems))
        for i,dp_em in enumerate(dp_ems):
            p_em_deflect = p_em*dp_em
            p4_em = [np.sqrt(m_e ** 2 + p_em_deflect ** 2), 0, p_em_deflect * np.sin(theta_em * np.pi / 180),
                     p_em_deflect * np.cos(theta_em * np.pi / 180)]
            p4_ep = [np.sqrt(m_e ** 2 + p_ep ** 2), 0, p_ep * np.sin(theta_ep * np.pi / 180),
                     p_ep * np.cos(theta_ep * np.pi / 180)]
            p4_jspi_recon = np.array(p4_em) + np.array(p4_ep)
            m_jspi_recon[i] = np.sqrt(
                p4_jspi_recon[0] ** 2 - p4_jspi_recon[1] ** 2 - p4_jspi_recon[2] ** 2 - p4_jspi_recon[3] ** 2)
        return m_jspi_recon

    # print(invMass_theta([0]))
    plt.subplot(2,1,1)
    dtheta_vals=np.linspace(-1,1,100)
    plt.plot(dtheta_vals,invMass_theta(dtheta_vals),label=r"$\theta_{em}$="+str(theta_em)+r"$^\circ$")


    plt.subplot(2,1,2)
    dp_em_vals=np.linspace(0.95,1,100)
    plt.plot(dp_em_vals,invMass_p_em(dp_em_vals),label=r"$\theta_{em}$="+str(theta_em)+r"$^\circ$")

plt.subplot(2,1,1)
plt.plot([-1,1],[m_jpsi,m_jpsi],'r--')
plt.xlabel(r"$\Delta$"+r" $\theta_{e^+}$ [Deg]")
plt.ylabel(r"J/$\psi$ mass")
plt.legend()

plt.subplot(2,1,2)
plt.plot([0.95,1],[m_jpsi,m_jpsi],'r--')
plt.xlabel(r"$p_{meas}/p_{true}$")
plt.ylabel(r"J/$\psi$ mass")
plt.show()