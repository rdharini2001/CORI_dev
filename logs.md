python disentangle.py.py 
After variance filtering: 1024
After correlation filtering: 946
After variance filtering: 1024
After correlation filtering: 149
Train event Task 1(cancer) counts: [14159  2156]
Train event Task 2(mace) counts: [14667  1648]
Test event Task 1(cancer) counts: [14763  2435]
Test event Task 2(mace) counts: [15402  1796]
[epoch 001] train | total=3.8993  task1=0.6709  task2=1.2254  orth1=7.7409  orth2=7.4796  recon=4.8154
[epoch 002] train | total=2.2734  task1=0.3595  task2=1.2011  orth1=1.4412  orth2=1.6617  recon=4.0586
[epoch 002] train | task1 AUC=0.949  task2 AUC=0.628  |  test  | task1 AUC=0.938  task2 AUC=0.609
[epoch 002] subgroup | train cancer AUC[MACE-]=0.957 [MACE+]=0.877  MACE AUC[cancer-]=0.624 [cancer+]=0.609  |  test  cancer AUC[MACE-]=0.946 [MACE+]=0.857  MACE AUC[cancer-]=0.601 [cancer+]=0.613
[epoch 002] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.639  |  P1->Y2(leak, want ~0.5)=0.602  P2->Y1(leak, want ~0.5)=0.842  |  cos(E1,P1)=33.493  cos(E1,P2)=30.140
[epoch 003] train | total=2.0557  task1=0.3263  task2=1.1929  orth1=1.1329  orth2=1.0194  recon=3.2770
[epoch 004] train | total=1.9550  task1=0.3183  task2=1.1911  orth1=0.9463  orth2=0.9142  recon=2.6895
[epoch 004] train | task1 AUC=0.952  task2 AUC=0.646  |  test  | task1 AUC=0.939  task2 AUC=0.616
[epoch 004] subgroup | train cancer AUC[MACE-]=0.960 [MACE+]=0.882  MACE AUC[cancer-]=0.646 [cancer+]=0.628  |  test  cancer AUC[MACE-]=0.948 [MACE+]=0.867  MACE AUC[cancer-]=0.609 [cancer+]=0.640
[epoch 004] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.641  |  P1->Y2(leak, want ~0.5)=0.595  P2->Y1(leak, want ~0.5)=0.833  |  cos(E1,P1)=31.385  cos(E1,P2)=31.654
[epoch 005] train | total=1.8637  task1=0.3102  task2=1.1869  orth1=0.8147  orth2=0.7512  recon=2.2233
[epoch 006] train | total=1.8222  task1=0.3115  task2=1.1852  orth1=0.7564  orth2=0.7902  recon=1.8589
[epoch 006] train | task1 AUC=0.955  task2 AUC=0.653  |  test  | task1 AUC=0.940  task2 AUC=0.620
[epoch 006] subgroup | train cancer AUC[MACE-]=0.963 [MACE+]=0.888  MACE AUC[cancer-]=0.653 [cancer+]=0.649  |  test  cancer AUC[MACE-]=0.948 [MACE+]=0.873  MACE AUC[cancer-]=0.614 [cancer+]=0.649
[epoch 006] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.642  |  P1->Y2(leak, want ~0.5)=0.597  P2->Y1(leak, want ~0.5)=0.812  |  cos(E1,P1)=27.997  cos(E1,P2)=24.562
[epoch 007] train | total=1.7591  task1=0.3086  task2=1.1788  orth1=0.6795  orth2=0.6344  recon=1.5790
[epoch 008] train | total=1.7456  task1=0.3062  task2=1.1783  orth1=0.7795  orth2=0.6431  recon=1.3925
[epoch 008] train | task1 AUC=0.957  task2 AUC=0.659  |  test  | task1 AUC=0.942  task2 AUC=0.625
[epoch 008] subgroup | train cancer AUC[MACE-]=0.965 [MACE+]=0.894  MACE AUC[cancer-]=0.662 [cancer+]=0.658  |  test  cancer AUC[MACE-]=0.949 [MACE+]=0.880  MACE AUC[cancer-]=0.620 [cancer+]=0.656
[epoch 008] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.646  |  P1->Y2(leak, want ~0.5)=0.585  P2->Y1(leak, want ~0.5)=0.856  |  cos(E1,P1)=25.903  cos(E1,P2)=26.888
[epoch 009] train | total=1.7307  task1=0.3050  task2=1.1734  orth1=0.8098  orth2=0.6764  recon=1.2654
[epoch 010] train | total=1.7047  task1=0.3043  task2=1.1684  orth1=0.6394  orth2=0.7620  recon=1.1723
[epoch 010] train | task1 AUC=0.958  task2 AUC=0.671  |  test  | task1 AUC=0.942  task2 AUC=0.632
[epoch 010] subgroup | train cancer AUC[MACE-]=0.966 [MACE+]=0.896  MACE AUC[cancer-]=0.677 [cancer+]=0.666  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.884  MACE AUC[cancer-]=0.628 [cancer+]=0.659
[epoch 010] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.647  |  P1->Y2(leak, want ~0.5)=0.588  P2->Y1(leak, want ~0.5)=0.885  |  cos(E1,P1)=20.562  cos(E1,P2)=20.907
[epoch 011] train | total=1.6704  task1=0.3008  task2=1.1711  orth1=0.5578  orth2=0.5970  recon=1.1112
[epoch 012] train | total=1.6655  task1=0.3037  task2=1.1616  orth1=0.6385  orth2=0.5979  recon=1.0672
[epoch 012] train | task1 AUC=0.961  task2 AUC=0.684  |  test  | task1 AUC=0.943  task2 AUC=0.638
[epoch 012] subgroup | train cancer AUC[MACE-]=0.968 [MACE+]=0.901  MACE AUC[cancer-]=0.691 [cancer+]=0.669  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.886  MACE AUC[cancer-]=0.635 [cancer+]=0.663
[epoch 012] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.648  |  P1->Y2(leak, want ~0.5)=0.590  P2->Y1(leak, want ~0.5)=0.846  |  cos(E1,P1)=26.858  cos(E1,P2)=20.005
[epoch 013] train | total=1.6558  task1=0.2996  task2=1.1570  orth1=0.6094  orth2=0.6763  recon=1.0337
[epoch 014] train | total=1.6421  task1=0.3010  task2=1.1534  orth1=0.5715  orth2=0.6530  recon=1.0015
[epoch 014] train | task1 AUC=0.961  task2 AUC=0.698  |  test  | task1 AUC=0.943  task2 AUC=0.646
[epoch 014] subgroup | train cancer AUC[MACE-]=0.969 [MACE+]=0.903  MACE AUC[cancer-]=0.703 [cancer+]=0.682  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.886  MACE AUC[cancer-]=0.642 [cancer+]=0.668
[epoch 014] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.649  |  P1->Y2(leak, want ~0.5)=0.581  P2->Y1(leak, want ~0.5)=0.737  |  cos(E1,P1)=19.998  cos(E1,P2)=18.390
[epoch 015] train | total=1.6106  task1=0.2988  task2=1.1465  orth1=0.5021  orth2=0.5460  recon=0.9772
[epoch 016] train | total=1.6111  task1=0.3001  task2=1.1387  orth1=0.6620  orth2=0.4991  recon=0.9552
[epoch 016] train | task1 AUC=0.963  task2 AUC=0.705  |  test  | task1 AUC=0.944  task2 AUC=0.648
[epoch 016] subgroup | train cancer AUC[MACE-]=0.970 [MACE+]=0.906  MACE AUC[cancer-]=0.710 [cancer+]=0.686  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.888  MACE AUC[cancer-]=0.642 [cancer+]=0.671
[epoch 016] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.650  |  P1->Y2(leak, want ~0.5)=0.599  P2->Y1(leak, want ~0.5)=0.751  |  cos(E1,P1)=23.158  cos(E1,P2)=21.584
[epoch 017] train | total=1.5976  task1=0.2987  task2=1.1372  orth1=0.5756  orth2=0.5243  recon=0.9330
[epoch 018] train | total=1.5846  task1=0.2962  task2=1.1342  orth1=0.5047  orth2=0.5609  recon=0.9167
[epoch 018] train | task1 AUC=0.963  task2 AUC=0.712  |  test  | task1 AUC=0.944  task2 AUC=0.654
[epoch 018] subgroup | train cancer AUC[MACE-]=0.971 [MACE+]=0.907  MACE AUC[cancer-]=0.720 [cancer+]=0.698  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.888  MACE AUC[cancer-]=0.651 [cancer+]=0.670
[epoch 018] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.651  |  P1->Y2(leak, want ~0.5)=0.584  P2->Y1(leak, want ~0.5)=0.809  |  cos(E1,P1)=16.948  cos(E1,P2)=16.357
[epoch 019] train | total=1.5878  task1=0.2958  task2=1.1245  orth1=0.6067  orth2=0.6317  recon=0.8949
[epoch 020] train | total=1.5723  task1=0.2968  task2=1.1225  orth1=0.5905  orth2=0.5368  recon=0.8826
[epoch 020] train | task1 AUC=0.965  task2 AUC=0.719  |  test  | task1 AUC=0.944  task2 AUC=0.649
[epoch 020] subgroup | train cancer AUC[MACE-]=0.972 [MACE+]=0.912  MACE AUC[cancer-]=0.718 [cancer+]=0.705  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.887  MACE AUC[cancer-]=0.639 [cancer+]=0.675
[epoch 020] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.651  |  P1->Y2(leak, want ~0.5)=0.586  P2->Y1(leak, want ~0.5)=0.805  |  cos(E1,P1)=17.176  cos(E1,P2)=17.665
[epoch 021] train | total=1.5672  task1=0.2968  task2=1.1248  orth1=0.4988  orth2=0.5994  recon=0.8622
[epoch 022] train | total=1.5470  task1=0.2944  task2=1.1195  orth1=0.4621  orth2=0.5504  recon=0.8440
[epoch 022] train | task1 AUC=0.966  task2 AUC=0.727  |  test  | task1 AUC=0.944  task2 AUC=0.652
[epoch 022] subgroup | train cancer AUC[MACE-]=0.973 [MACE+]=0.914  MACE AUC[cancer-]=0.726 [cancer+]=0.722  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.889  MACE AUC[cancer-]=0.641 [cancer+]=0.674
[epoch 022] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.652  |  P1->Y2(leak, want ~0.5)=0.595  P2->Y1(leak, want ~0.5)=0.839  |  cos(E1,P1)=21.273  cos(E1,P2)=22.742
[epoch 023] train | total=1.5375  task1=0.2940  task2=1.1094  orth1=0.5153  orth2=0.5434  recon=0.8264
[epoch 024] train | total=1.5203  task1=0.2957  task2=1.1086  orth1=0.4288  orth2=0.4811  recon=0.8144
[epoch 024] train | task1 AUC=0.966  task2 AUC=0.725  |  test  | task1 AUC=0.944  task2 AUC=0.642
[epoch 024] subgroup | train cancer AUC[MACE-]=0.973 [MACE+]=0.914  MACE AUC[cancer-]=0.721 [cancer+]=0.725  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.892  MACE AUC[cancer-]=0.629 [cancer+]=0.677
[epoch 024] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.650  |  P1->Y2(leak, want ~0.5)=0.581  P2->Y1(leak, want ~0.5)=0.789  |  cos(E1,P1)=16.339  cos(E1,P2)=24.194
[epoch 025] train | total=1.5193  task1=0.2920  task2=1.1102  orth1=0.4767  orth2=0.4872  recon=0.7964
[epoch 026] train | total=1.5026  task1=0.2910  task2=1.0961  orth1=0.5316  orth2=0.4488  recon=0.7786
[epoch 026] train | task1 AUC=0.968  task2 AUC=0.737  |  test  | task1 AUC=0.944  task2 AUC=0.654
[epoch 026] subgroup | train cancer AUC[MACE-]=0.974 [MACE+]=0.922  MACE AUC[cancer-]=0.741 [cancer+]=0.754  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.891  MACE AUC[cancer-]=0.650 [cancer+]=0.668
[epoch 026] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.650  |  P1->Y2(leak, want ~0.5)=0.587  P2->Y1(leak, want ~0.5)=0.863  |  cos(E1,P1)=21.541  cos(E1,P2)=21.203
[epoch 027] train | total=1.4964  task1=0.2872  task2=1.0997  orth1=0.4985  orth2=0.4590  recon=0.7688
[epoch 028] train | total=1.4756  task1=0.2894  task2=1.0861  orth1=0.4585  orth2=0.4317  recon=0.7554
[epoch 028] train | task1 AUC=0.968  task2 AUC=0.740  |  test  | task1 AUC=0.944  task2 AUC=0.656
[epoch 028] subgroup | train cancer AUC[MACE-]=0.974 [MACE+]=0.925  MACE AUC[cancer-]=0.747 [cancer+]=0.760  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.888  MACE AUC[cancer-]=0.654 [cancer+]=0.668
[epoch 028] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.648  |  P1->Y2(leak, want ~0.5)=0.597  P2->Y1(leak, want ~0.5)=0.784  |  cos(E1,P1)=20.752  cos(E1,P2)=20.396
[epoch 029] train | total=1.4718  task1=0.2892  task2=1.0903  orth1=0.3806  orth2=0.4726  recon=0.7386
[epoch 030] train | total=1.4795  task1=0.2883  task2=1.0816  orth1=0.4592  orth2=0.5923  recon=0.7312
[epoch 030] train | task1 AUC=0.970  task2 AUC=0.753  |  test  | task1 AUC=0.944  task2 AUC=0.654
[epoch 030] subgroup | train cancer AUC[MACE-]=0.976 [MACE+]=0.927  MACE AUC[cancer-]=0.755 [cancer+]=0.779  |  test  cancer AUC[MACE-]=0.951 [MACE+]=0.885  MACE AUC[cancer-]=0.650 [cancer+]=0.653
[epoch 030] disentanglement | E1->Y1(sanity, want high)=0.945  E1->Y2(sanity, want high)=0.647  |  P1->Y2(leak, want ~0.5)=0.576  P2->Y1(leak, want ~0.5)=0.757  |  cos(E1,P1)=18.606  cos(E1,P2)=18.473
[epoch 031] train | total=1.4488  task1=0.2874  task2=1.0818  orth1=0.3244  orth2=0.4622  recon=0.7177
[epoch 032] train | total=1.4424  task1=0.2855  task2=1.0753  orth1=0.4111  orth2=0.4294  recon=0.7044
[epoch 032] train | task1 AUC=0.972  task2 AUC=0.759  |  test  | task1 AUC=0.944  task2 AUC=0.652
[epoch 032] subgroup | train cancer AUC[MACE-]=0.977 [MACE+]=0.932  MACE AUC[cancer-]=0.758 [cancer+]=0.792  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.886  MACE AUC[cancer-]=0.645 [cancer+]=0.660
[epoch 032] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.644  |  P1->Y2(leak, want ~0.5)=0.591  P2->Y1(leak, want ~0.5)=0.870  |  cos(E1,P1)=20.250  cos(E1,P2)=22.093
[epoch 033] train | total=1.4268  task1=0.2830  task2=1.0623  orth1=0.4239  orth2=0.4418  recon=0.6955
[epoch 034] train | total=1.4491  task1=0.2838  task2=1.0755  orth1=0.4818  orth2=0.5027  recon=0.6864
[epoch 034] train | task1 AUC=0.972  task2 AUC=0.745  |  test  | task1 AUC=0.943  task2 AUC=0.655
[epoch 034] subgroup | train cancer AUC[MACE-]=0.978 [MACE+]=0.933  MACE AUC[cancer-]=0.756 [cancer+]=0.802  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.885  MACE AUC[cancer-]=0.657 [cancer+]=0.661
[epoch 034] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.643  |  P1->Y2(leak, want ~0.5)=0.583  P2->Y1(leak, want ~0.5)=0.874  |  cos(E1,P1)=18.456  cos(E1,P2)=17.377
[epoch 035] train | total=1.4360  task1=0.2803  task2=1.0650  orth1=0.4901  orth2=0.5364  recon=0.6734
[epoch 036] train | total=1.4237  task1=0.2810  task2=1.0605  orth1=0.4779  orth2=0.4910  recon=0.6650
[epoch 036] train | task1 AUC=0.974  task2 AUC=0.768  |  test  | task1 AUC=0.944  task2 AUC=0.650
[epoch 036] subgroup | train cancer AUC[MACE-]=0.979 [MACE+]=0.938  MACE AUC[cancer-]=0.767 [cancer+]=0.816  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.888  MACE AUC[cancer-]=0.646 [cancer+]=0.649
[epoch 036] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.638  |  P1->Y2(leak, want ~0.5)=0.575  P2->Y1(leak, want ~0.5)=0.818  |  cos(E1,P1)=24.033  cos(E1,P2)=20.342
[epoch 037] train | total=1.4227  task1=0.2793  task2=1.0447  orth1=0.5390  orth2=0.6176  recon=0.6563
[epoch 038] train | total=1.4520  task1=0.2770  task2=1.0537  orth1=0.6321  orth2=0.7867  recon=0.6495
[epoch 038] train | task1 AUC=0.975  task2 AUC=0.783  |  test  | task1 AUC=0.943  task2 AUC=0.647
[epoch 038] subgroup | train cancer AUC[MACE-]=0.980 [MACE+]=0.940  MACE AUC[cancer-]=0.775 [cancer+]=0.831  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.886  MACE AUC[cancer-]=0.636 [cancer+]=0.665
[epoch 038] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.637  |  P1->Y2(leak, want ~0.5)=0.584  P2->Y1(leak, want ~0.5)=0.879  |  cos(E1,P1)=18.771  cos(E1,P2)=20.012
[epoch 039] train | total=1.3707  task1=0.2735  task2=1.0374  orth1=0.3579  orth2=0.4697  recon=0.6423
[epoch 040] train | total=1.3613  task1=0.2719  task2=1.0305  orth1=0.4036  orth2=0.4448  recon=0.6316
[epoch 040] train | task1 AUC=0.977  task2 AUC=0.786  |  test  | task1 AUC=0.943  task2 AUC=0.639
[epoch 040] subgroup | train cancer AUC[MACE-]=0.981 [MACE+]=0.943  MACE AUC[cancer-]=0.778 [cancer+]=0.833  |  test  cancer AUC[MACE-]=0.949 [MACE+]=0.882  MACE AUC[cancer-]=0.628 [cancer+]=0.657
[epoch 040] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.636  |  P1->Y2(leak, want ~0.5)=0.577  P2->Y1(leak, want ~0.5)=0.813  |  cos(E1,P1)=19.049  cos(E1,P2)=20.924
[epoch 041] train | total=1.3504  task1=0.2709  task2=1.0222  orth1=0.4085  orth2=0.4535  recon=0.6219
[epoch 042] train | total=1.3688  task1=0.2726  task2=1.0155  orth1=0.5651  orth2=0.5508  recon=0.6177
[epoch 042] train | task1 AUC=0.978  task2 AUC=0.793  |  test  | task1 AUC=0.943  task2 AUC=0.621
[epoch 042] subgroup | train cancer AUC[MACE-]=0.982 [MACE+]=0.945  MACE AUC[cancer-]=0.778 [cancer+]=0.848  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.884  MACE AUC[cancer-]=0.610 [cancer+]=0.650
[epoch 042] disentanglement | E1->Y1(sanity, want high)=0.944  E1->Y2(sanity, want high)=0.632  |  P1->Y2(leak, want ~0.5)=0.579  P2->Y1(leak, want ~0.5)=0.805  |  cos(E1,P1)=24.619  cos(E1,P2)=21.155
[epoch 043] train | total=1.3430  task1=0.2683  task2=1.0262  orth1=0.3929  orth2=0.4446  recon=0.6081
[epoch 044] train | total=1.3211  task1=0.2634  task2=1.0166  orth1=0.3604  orth2=0.4352  recon=0.5998
[epoch 044] train | task1 AUC=0.977  task2 AUC=0.788  |  test  | task1 AUC=0.943  task2 AUC=0.651
[epoch 044] subgroup | train cancer AUC[MACE-]=0.982 [MACE+]=0.944  MACE AUC[cancer-]=0.787 [cancer+]=0.858  |  test  cancer AUC[MACE-]=0.950 [MACE+]=0.887  MACE AUC[cancer-]=0.645 [cancer+]=0.653
[epoch 044] disentanglement | E1->Y1(sanity, want high)=0.943  E1->Y2(sanity, want high)=0.630  |  P1->Y2(leak, want ~0.5)=0.579  P2->Y1(leak, want ~0.5)=0.718  |  cos(E1,P1)=15.881  cos(E1,P2)=17.905
[epoch 045] train | total=1.3250  task1=0.2659  task2=1.0225  orth1=0.3475  orth2=0.4283  recon=0.5968
[epoch 046] train | total=1.3178  task1=0.2620  task2=1.0136  orth1=0.3991  orth2=0.4595  recon=0.5921
[epoch 046] train | task1 AUC=0.981  task2 AUC=0.810  |  test  | task1 AUC=0.942  task2 AUC=0.621
[epoch 046] subgroup | train cancer AUC[MACE-]=0.984 [MACE+]=0.952  MACE AUC[cancer-]=0.793 [cancer+]=0.879  |  test  cancer AUC[MACE-]=0.949 [MACE+]=0.879  MACE AUC[cancer-]=0.608 [cancer+]=0.650
[epoch 046] disentanglement | E1->Y1(sanity, want high)=0.943  E1->Y2(sanity, want high)=0.625  |  P1->Y2(leak, want ~0.5)=0.593  P2->Y1(leak, want ~0.5)=0.810  |  cos(E1,P1)=18.989  cos(E1,P2)=14.599
[epoch 047] train | total=1.3309  task1=0.2613  task2=1.0101  orth1=0.5113  orth2=0.5549  recon=0.5808
[epoch 048] train | total=1.2937  task1=0.2589  task2=0.9996  orth1=0.4061  orth2=0.4423  recon=0.5765
[epoch 048] train | task1 AUC=0.982  task2 AUC=0.808  |  test  | task1 AUC=0.942  task2 AUC=0.629
[epoch 048] subgroup | train cancer AUC[MACE-]=0.985 [MACE+]=0.956  MACE AUC[cancer-]=0.792 [cancer+]=0.880  |  test  cancer AUC[MACE-]=0.949 [MACE+]=0.884  MACE AUC[cancer-]=0.621 [cancer+]=0.634
[epoch 048] disentanglement | E1->Y1(sanity, want high)=0.943  E1->Y2(sanity, want high)=0.621  |  P1->Y2(leak, want ~0.5)=0.560  P2->Y1(leak, want ~0.5)=0.826  |  cos(E1,P1)=14.829  cos(E1,P2)=16.634
[epoch 049] train | total=1.2936  task1=0.2513  task2=1.0064  orth1=0.4251  orth2=0.4720  recon=0.5732
[epoch 050] train | total=1.2651  task1=0.2461  task2=0.9906  orth1=0.4141  orth2=0.4397  recon=0.5671
[epoch 050] train | task1 AUC=0.982  task2 AUC=0.823  |  test  | task1 AUC=0.943  task2 AUC=0.626
[epoch 050] subgroup | train cancer AUC[MACE-]=0.985 [MACE+]=0.955  MACE AUC[cancer-]=0.807 [cancer+]=0.909  |  test  cancer AUC[MACE-]=0.949 [MACE+]=0.883  MACE AUC[cancer-]=0.616 [cancer+]=0.631
[epoch 050] disentanglement | E1->Y1(sanity, want high)=0.943  E1->Y2(sanity, want high)=0.619  |  P1->Y2(leak, want ~0.5)=0.573  P2->Y1(leak, want ~0.5)=0.797  |  cos(E1,P1)=16.310  cos(E1,P2)=16.029
(hest) saitama@somai-l-345b:/mnt/c/Users/ap4si/OneDrive - Emory/Data/Shared_data/people/Dharini_Raghavan/CORI/CORI_dev_code$ 