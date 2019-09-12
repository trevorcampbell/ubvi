import numpy as np
from scipy.stats import norm, multivariate_normal
import unittest
from distributions import Gaussian

class TestGaussian(unittest.TestCase):
    
    def setUp(self):
        self.g1f = Gaussian(1, False)
        self.g2f = Gaussian(2, False)
        self.g4t = Gaussian(4, True)
        self.param1f = np.array([0,2])
        self.param2f = np.array([np.arange(6), 2*np.arange(6)])
        self.param4t = np.array([np.arange(8), -np.arange(8)])
       
    def test_reparam(self):
        t1f = {"g_mu":np.array([0]), "g_Sig":np.array([4]), "g_Siginv":np.array([0.25])}
        theta1f = self.g1f.reparam(self.param1f)
        self.assertDictEqual(t1f, theta1f)
        
        t2f = {"g_mu":np.array([[0,1],[0,2]]), "g_Sig":np.array([[[13,23],[23,41]],[[52,92],[92,164]]]), "g_Siginv":np.array([[[10.25,-5.75],[-5.75,3.25]],[[2.5625,-1.4375],[-1.4375,0.8125]]])}
        theta2f = self.g2f.reparam(self.param2f)
        for key in t2f.keys():
            with self.subTest(key=key):
                self.assertTrue(np.all(np.round(t2f[key],2)==np.round(theta2f[key],2)))
        
        t4t = {"g_mu":np.array([[0,1,2,3],[0,-1,-2,-3]]), "g_Sig":np.array([np.exp([4,5,6,7]),np.exp([-4,-5,-6,-7])])}
        theta4t = self.g4t.reparam(self.param4t)
        for key in t4t.keys():
            with self.subTest(key=key):
                self.assertTrue(np.all(t4t[key]==theta4t[key]))
    
    def test_logpdf(self):
        X1f = np.random.randn(5)
        p1f = norm.logpdf(X1f, 0, 2)
        logp1f = self.g1f.logpdf(self.param1f, X1f)
        self.assertTrue(np.all(np.round(p1f[:,np.newaxis],5)==np.round(logp1f,5)))
        
        X2f = np.random.randn(2)
        p2f1 = multivariate_normal.logpdf(X2f, np.array([0,1]), np.array([[13,23],[23,41]]))
        p2f2 = multivariate_normal.logpdf(X2f, np.array([0,2]), np.array([[52,92],[92,164]]))
        p2f = np.array([p2f1, p2f2])
        logp2f = self.g2f.logpdf(self.param2f, X2f)
        self.assertTrue(np.all(np.round(p2f[np.newaxis,:],5)==np.round(logp2f,5)))
        
        X4t = np.random.randn(8).reshape((2,4))
        p4t1 = multivariate_normal.logpdf(X4t, np.array([0,1,2,3]),np.diag(np.exp(np.array([4,5,6,7]))))
        p4t2 = multivariate_normal.logpdf(X4t, np.array([0,-1,-2,-3]),np.diag(np.exp(np.array([-4,-5,-6,-7]))))
        p4t = np.hstack((p4t1[:,np.newaxis], p4t2[:,np.newaxis]))
        logp4t = self.g4t.logpdf(self.param4t, X4t)
        self.assertTrue(np.all(np.around(p4t,5)==np.around(logp4t,5)))
    
    def test_sample(self):
        np.random.seed(1)
        x1 = 0 + 2* np.random.randn(5)
        x2 = np.array([0,1]) + np.dot(np.random.randn(1,2), np.array([[2,3],[4,5]]))
        x4 = np.array([0,1,2,3]) + np.dot(np.random.randn(10,4), np.diag(np.exp(np.array([4,5,6,7])/2)))
        
        np.random.seed(1)
        samp1 = self.g1f.sample(self.param1f, 5)
        samp2 = self.g2f.sample(self.param2f[0,:], 1)
        samp4 = self.g4t.sample(self.param4t[0,:], 10)
        
        self.assertTrue(np.all(np.round(x1[:,np.newaxis],3)==np.round(samp1,3)))
        self.assertTrue(np.all(np.round(x2,3)==np.round(samp2,3)))
        self.assertTrue(np.all(np.round(x4,3)==np.round(samp4,3)))
    
    def test_cross_sample(self):
        var1f = 2 / np.array([1/4+1/25])
        mu1f = np.array([0])
        cov2f = 2 * np.linalg.inv(np.array([[10.25,-5.75],[-5.75,3.25]])+np.array([[2.5625,-1.4375],[-1.4375,0.8125]]))
        mu2f = 0.5*np.dot(cov2f, np.dot(np.array([[10.25,-5.75],[-5.75,3.25]]),np.array([0,1])) + np.dot(np.array([[2.5625,-1.4375],[-1.4375,0.8125]]),np.array([0,2])))
        cov4t = 2 /(np.exp(np.arange(-4,-8,-1)) + np.exp(np.arange(4,8)))
        mu4t = 0.5*(cov4t*(np.arange(4)*np.exp(np.arange(-4,-8,-1))+np.arange(0,-4,-1)*np.exp(np.arange(4,8))))               
        np.random.seed(1)
        x1 = np.random.multivariate_normal(mu1f, np.atleast_2d(var1f), 5)
        np.random.seed(2)
        x2 = np.random.multivariate_normal(mu2f, cov2f, 1)
        np.random.seed(0)
        x4 = mu4t + np.sqrt(cov4t)*np.random.randn(10, 4)
        
        np.random.seed(1)
        samp1 = self.g1f.cross_sample(self.param1f, -2.5*self.param1f, 5)
        np.random.seed(2)
        samp2 = self.g2f.cross_sample(self.param2f[0,:], self.param2f[1,:], 1)
        np.random.seed(0)
        samp4 = self.g4t.cross_sample(self.param4t[0,:], self.param4t[1,:], 10)
        
        self.assertTrue(np.all(np.round(x1,3)==np.round(samp1,3)))
        self.assertTrue(np.all(np.round(x2,3)==np.round(samp2,3)))
        self.assertTrue(np.all(np.round(x4,3)==np.round(samp4,3)))
    
    def test_log_sqrt_pair_integral(self):
        l1 = -0.5*np.log(10/(2*4))
        difmu2 = np.array([0,1])
        s = np.array([[13,23],[23,41]])
        S = np.array([[52,92],[92,164]])
        S2 = np.array([[32.5,57.5],[57.5,102.5]])
        l21 = -0.125*(difmu2*np.linalg.solve(S2,difmu2)).sum()- 0.5*np.linalg.slogdet(S2)[1] + 0.25*np.linalg.slogdet(s)[1] + 0.25*np.linalg.slogdet(S)[1]
        l2 = np.array([0,l21])
        difmu4 = np.array([[0,-0.5,-1,-1.5],[0,2.5,5,7.5]])
        lsig = self.param4t[0, 4:]
        Lsig = self.param4t[:,4:]*1.5
        lSig2 = np.log(0.5)+np.logaddexp(lsig, Lsig)
        l4 = -0.125*np.sum(np.exp(-lSig2)*difmu4**2, axis=1) - 0.5*np.sum(lSig2, axis=1) + 0.25*np.sum(lsig) + 0.25*np.sum(Lsig, axis=1)
        
        la1f = self.g1f.log_sqrt_pair_integral(self.param1f, -2*self.param1f)
        la2f = self.g2f.log_sqrt_pair_integral(self.param2f[0,:], self.param2f)
        la4t = self.g4t.log_sqrt_pair_integral(self.param4t[0,:], 1.5*self.param4t)
        
        self.assertTrue(np.all(np.round(l1,5)==np.round(la1f,5)))
        self.assertTrue(np.all(np.round(l2,5)==np.round(la2f,5)))
        self.assertTrue(np.all(np.round(l4,5)==np.round(la4t,5)))
    
    def test_params_init(self):
        np.random.seed(1)
        m0 = np.random.multivariate_normal(np.zeros(2), 4*np.eye(2))
        prm0f = np.concatenate((m0, np.array([1,0,0,1])))
        np.random.seed(2)
        m0 = np.random.multivariate_normal(np.zeros(2), 4*np.eye(2))
        prm0t = np.concatenate((m0, np.zeros(2)))
        np.random.seed(4)
        mu = np.array([[0,1,2,3],[0,-1,-2,-3]])
        k = np.random.choice(2, p=np.array([0.5,0.5]))
        lsig = np.array([[4,5,6,7],[-4,-5,-6,-7]])
        mu0 = mu[k]+np.random.randn(4)*np.sqrt(10)*np.exp(lsig[k,:])
        LSig = np.random.randn(4)+lsig[k] 
        prm4t = np.hstack((mu0, LSig))
        
        g2t = Gaussian(2, True)
        np.random.seed(1)
        par0f = self.g2f.params_init(np.empty((0,6)), np.empty((0,1)), 4)
        np.random.seed(2)
        par0t = g2t.params_init(np.empty((0,6)), np.empty((0,1)), 4)
        np.random.seed(4)
        par4t = self.g4t.params_init(self.param4t, np.array([0.1, 0.9]), 10)
        
        self.assertTrue(np.all(np.round(par0f,3)==np.round(prm0f,3)))
        self.assertTrue(np.all(np.round(par0t,3)==np.round(prm0t,3)))
        self.assertTrue(np.all(np.round(par4t,3)==np.round(prm4t,3)))

if __name__ == "__main__":
    unittest.main()
    