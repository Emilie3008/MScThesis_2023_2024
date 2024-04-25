
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.31' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jul 14 2019 19:59:36' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 18])  = 'FC_Tf_1073_Tc_1073' ;
WORKING_DIRECTORY         (idx, [1: 48])  = '/home/aimetta/GA_SCK/Tf_1073_Tc_1073/GPT_reduced' ;
HOSTNAME                  (idx, [1: 11])  = 'blanketcool' ;
CPU_TYPE                  (idx, [1: 35])  = 'Intel Xeon Processor (Skylake, IBRS' ;
CPU_MHZ                   (idx, 1)        = 1.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Feb 20 08:16:04 2024' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Thu Feb 22 06:43:27 2024' ;

% Run parameters:

POP                       (idx, 1)        = 1000000 ;
CYCLES                    (idx, 1)        = 1000 ;
SKIP                      (idx, 1)        = 50 ;
BATCH_INTERVAL            (idx, 1)        = 25 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1708416964902 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 0 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 1 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 0 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 0 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 1 ;
OMP_THREADS               (idx, 1)        = 30 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  30]) = [  9.78509E-01  1.06572E+00  9.48832E-01  1.01819E+00  9.30865E-01  9.66500E-01  9.84256E-01  1.01568E+00  1.04849E+00  9.90588E-01  9.46743E-01  1.04080E+00  1.01255E+00  1.03342E+00  9.88397E-01  1.09710E+00  8.83356E-01  9.71538E-01  1.01458E+00  9.95424E-01  9.69428E-01  9.98128E-01  1.02164E+00  1.02533E+00  9.14950E-01  1.10434E+00  9.77515E-01  1.06194E+00  9.95972E-01  9.99217E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 45])  = '/opt/serpent/xsdata/endfb8/sss2_endfb8.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1:  3])  = 'N/A' ;
SFY_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;
NFY_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  9.72616E-02 0.00014  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.02738E-01 1.5E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  2.93954E-01 1.4E-05  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  1.54514E-01 9.1E-06  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  7.63483E+00 5.9E-05  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  5.91765E+03 0.00060  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  5.91690E+03 0.00060  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  1.32299E+04 0.00060  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.58972E+03 0.00063  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 300 ;
SIMULATED_HISTORIES       (idx, 1)        = 300013811 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  1.00005E+06 0.00011 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  1.00005E+06 0.00011 ];
SIMULATION_COMPLETED      (idx, 1)        = 0 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.33684E+04 ;
RUNNING_TIME              (idx, 1)        =  2.78737E+03 ;
INIT_TIME                 (idx, [1:  2])  = [  1.32033E+00  1.32033E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.69000E-02  2.69000E-02 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.78603E+03  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  8.78681E+03  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 19.14648 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  1.91410E+01 0.00180 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  6.46177E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 128827.30 ;
ALLOC_MEMSIZE             (idx, 1)        = 77920.73;
MEMSIZE                   (idx, 1)        = 77257.76;
XS_MEMSIZE                (idx, 1)        = 1473.72;
MAT_MEMSIZE               (idx, 1)        = 1.45;
RES_MEMSIZE               (idx, 1)        = 13254.51;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 26704.51;
UNKNOWN_MEMSIZE           (idx, 1)        = 35823.57;
UNUSED_MEMSIZE            (idx, 1)        = 662.97;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 200 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1042278 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-11 ;
NEUTRON_EMAX              (idx, 1)        =  2.00000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.50000E-04 ;
URES_EMAX                 (idx, 1)        =  3.00000E+00 ;
URES_AVAIL                (idx, 1)        = 31 ;
URES_USED                 (idx, 1)        = 31 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 76 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 76 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 0 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 2408 ;
TOT_TRANSMU_REA           (idx, 1)        = 0 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  0.00000E+00 ;
TOT_DECAY_HEAT            (idx, 1)        =  0.00000E+00 ;
TOT_SF_RATE               (idx, 1)        =  0.00000E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  0.00000E+00 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  0.00000E+00 ;
INGESTION_TOXICITY        (idx, 1)        =  0.00000E+00 ;
ACTINIDE_INH_TOX          (idx, 1)        =  0.00000E+00 ;
ACTINIDE_ING_TOX          (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  0.00000E+00 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  0.00000E+00 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  0.00000E+00 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  3.99080E-08 1.7E-06  0.00000E+00 0.0E+00 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  6.39346E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  5.73696E-03 0.00092  1.57061E-02 0.00089 ];
U238_FISS                 (idx, [1:   4]) = [  3.08772E-02 0.00027  8.45325E-02 0.00019 ];
PU239_FISS                (idx, [1:   4]) = [  2.52568E-01 0.00012  6.91455E-01 4.1E-05 ];
PU240_FISS                (idx, [1:   4]) = [  2.63280E-02 0.00026  7.20783E-02 0.00030 ];
PU241_FISS                (idx, [1:   4]) = [  3.71622E-02 0.00019  1.01739E-01 0.00018 ];
U235_CAPT                 (idx, [1:   4]) = [  1.61337E-03 0.00174  2.66937E-03 0.00172 ];
U238_CAPT                 (idx, [1:   4]) = [  2.00837E-01 0.00015  3.32292E-01 0.00012 ];
PU239_CAPT                (idx, [1:   4]) = [  6.62269E-02 0.00018  1.09575E-01 0.00016 ];
PU240_CAPT                (idx, [1:   4]) = [  3.19284E-02 0.00040  5.28267E-02 0.00035 ];
PU241_CAPT                (idx, [1:   4]) = [  6.38577E-03 0.00080  1.05655E-02 0.00083 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 300013811 3.00000E+08 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 6.91643E+05 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 300013811 3.00692E+08 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 181300675 1.81738E+08 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 109611787 1.09834E+08 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 9101349 9.12025E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 300013811 3.00692E+08 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.86060E-03 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   2]) = [  1.21762E-11 0.00012 ];
TOT_POWDENS               (idx, [1:   2]) = [  9.99391E-20 0.00012 ];
TOT_GENRATE               (idx, [1:   2]) = [  1.07046E+00 0.00012 ];
TOT_FISSRATE              (idx, [1:   2]) = [  3.65270E-01 0.00012 ];
TOT_CAPTRATE              (idx, [1:   2]) = [  6.04399E-01 7.4E-05 ];
TOT_ABSRATE               (idx, [1:   2]) = [  9.69669E-01 1.3E-05 ];
TOT_SRCRATE               (idx, [1:   2]) = [  9.97700E-01 1.7E-06 ];
TOT_FLUX                  (idx, [1:   2]) = [  3.58559E+02 6.1E-05 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  3.03309E-02 0.00042 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  1.00000E+00 0.0E+00 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.18456E+02 7.1E-05 ];
INI_FMASS                 (idx, 1)        =  1.21836E+02 ;
TOT_FMASS                 (idx, 1)        =  1.21836E+02 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.42807E+00 0.00540 ];
SIX_FF_F                  (idx, [1:   2]) = [  2.13345E-02 0.00586 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.72583E-03 0.00079 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.75542E+03 0.00836 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.69702E-01 1.3E-05 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99894E-01 4.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  1.10657E+00 0.00012 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  1.07293E+00 0.00012 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.93060E+00 8.3E-06 ];
FISSE                     (idx, [1:   2]) = [  2.08059E+02 6.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  1.07292E+00 0.00011  2.67358E+01 0.00012  8.73687E-02 0.00156 ];
IMP_KEFF                  (idx, [1:   2]) = [  1.07293E+00 0.00012 ];
COL_KEFF                  (idx, [1:   2]) = [  1.07293E+00 0.00012 ];
ABS_KEFF                  (idx, [1:   2]) = [  1.07293E+00 0.00012 ];
ABS_KINF                  (idx, [1:   2]) = [  1.10658E+00 0.00012 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 2.000000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  5.01753E+00 4.8E-05 ];
IMP_ALF                   (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32418E-01 0.00024 ];
IMP_EALF                  (idx, [1:   2]) = [  2.00000E+01 0.0E+00 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  6.98649E-01 0.00019 ];
IMP_AFGE                  (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49969E-03 0.00076  8.99854E-05 0.00699  6.76072E-04 0.00212  5.39521E-04 0.00238  1.22810E-03 0.00181  7.13676E-04 0.00217  2.52342E-04 0.00417 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  5.22865E-01 0.00159  1.33813E-02 6.9E-05  3.08085E-02 4.3E-05  1.17000E-01 9.0E-05  3.06735E-01 6.4E-05  8.78136E-01 5.0E-05  2.93769E+00 0.00012 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 15 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  6.70894E-07 0.00085  6.70275E-07 0.00085  8.59999E-07 0.00836 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.20130E-07 0.00054  7.19467E-07 0.00054  9.23109E-07 0.00827 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.25695E-03 0.00185  8.31097E-05 0.00972  6.31893E-04 0.00375  4.92484E-04 0.00309  1.15059E-03 0.00223  6.62332E-04 0.00515  2.36547E-04 0.00526 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  5.24465E-01 0.00210  1.33844E-02 0.00016  3.08118E-02 4.8E-05  1.17038E-01 0.00017  3.06885E-01 0.00015  8.78249E-01 0.00016  2.93875E+00 0.00030 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  6.33140E-07 0.00181  6.32422E-07 0.00174  8.53075E-07 0.03554 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.79604E-07 0.00153  6.78833E-07 0.00146  9.15639E-07 0.03539 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.24672E-03 0.00518  8.07122E-05 0.05094  6.19645E-04 0.01392  4.95745E-04 0.01262  1.15503E-03 0.01200  6.59252E-04 0.00915  2.36342E-04 0.01971 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.25830E-01 0.00884  1.33808E-02 0.00046  3.08080E-02 0.00016  1.16978E-01 0.00067  3.07050E-01 0.00033  8.78529E-01 0.00030  2.94037E+00 0.00057 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  6.52752E-07 0.00108 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.00657E-07 0.00075 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.25690E-03 0.00230 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.98962E+03 0.00298 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  1.45642E-08 0.00022 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  1.62196E-04 0.00023  1.62196E-04 0.00023  1.62202E-04 0.00325 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.69329E-05 0.00116  7.69326E-05 0.00114  7.71168E-05 0.01859 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.16065E-03 0.00069  4.16032E-03 0.00067  4.25494E-03 0.00864 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.09170E+01 0.00224 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  2.36903E+02 7.0E-05  1.11181E+02 0.00012 ];

