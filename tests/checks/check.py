import os

from tests.checks import checks
from tests.harness import harness
from scripts import terminal

check_dir = os.path.dirname(os.path.abspath(__file__))

check_suite = checks.CheckSuite()

folds = [3]
inf_folds = [3]
incs = [1]

FLT_BIN_WIDTH=13
FLT_MAX_EXP=128
FLT_BIG_EXP=13
FLT_SMALL_EXP=-12
FLT_MIN_EXP=-125
FLT_MANT_DIG=24
FLT_ONES = 0
for i in range(FLT_MANT_DIG):
  FLT_ONES += 2.0 ** -i

DBL_BIN_WIDTH=41
DBL_MAX_EXP=1024
DBL_BIG_EXP=27
DBL_SMALL_EXP=-27
DBL_MIN_EXP=-1021
DBL_MANT_DIG=53
DBL_ONES = 0
for i in range(DBL_MANT_DIG):
  DBL_ONES += 2.0 ** -i

check_suite.add_checks([checks.ValidateInternalDSCALETest(),\
                        checks.ValidateInternalSSCALETest()],\
                       ["N", "incX"],\
                       [[4], [1]])

check_suite.add_checks([checks.ValidateInternalUFPTest(),\
                        checks.ValidateInternalUFPFTest()],\
                       ["N", "incX"],\
                       [[10], [1, 2, 4]])

check_suite.add_checks([checks.ValidateInternalDINDEXTest(),\
                        checks.ValidateInternalSINDEXTest(),\
                        checks.ValidateInternalDMINDEXTest(),\
                        checks.ValidateInternalSMINDEXTest()],\
                       ["N", "incX"],\
                       [[4], [1]])

check_suite.add_checks([checks.ValidateInternalDAMAXTest(),\
                        checks.ValidateInternalZAMAXTest(),\
                        checks.ValidateInternalSAMAXTest(),\
                        checks.ValidateInternalCAMAXTest()],\
                       ["N", "incX"],\
                       [[4095], [1, 2, 4]])

check_suite.add_checks([checks.ValidateInternalRDSUMTest(),\
                        checks.ValidateInternalDBDBADDTest(),\
                        checks.ValidateInternalDIDADDTest(),\
                        checks.ValidateInternalDIDDEPOSITTest(),\
                        checks.ValidateInternalRSSUMTest(),\
                        checks.ValidateInternalSBSBADDTest(),\
                        checks.ValidateInternalSISADDTest(),\
                        checks.ValidateInternalSISDEPOSITTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "FillX"],\
                       [[4095], folds, incs, [1.0, -1.0],\
                        ["constant",\
                         "mountain",\
                         "+big",\
                         "++big",\
                         "+-big",\
                         "sine"]])

check_suite.add_checks([checks.ValidateInternalRZSUMTest(),\
                        checks.ValidateInternalZBZBADDTest(),\
                        checks.ValidateInternalZIZADDTest(),\
                        checks.ValidateInternalZIZDEPOSITTest(),\
                        checks.ValidateInternalRCSUMTest(),\
                        checks.ValidateInternalCBCBADDTest(),\
                        checks.ValidateInternalCICADDTest(),\
                        checks.ValidateInternalCICDEPOSITTest()],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX"],\
                       [[4095], folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        ["constant",\
                         "mountain",\
                         "+big",\
                         "++big",\
                         "+-big",\
                         "sine"]])

check_suite.add_checks([checks.ValidateInternalRDNRM2Test(),\
                        checks.ValidateInternalRDASUMTest(),\
                        checks.ValidateInternalRSNRM2Test(),\
                        checks.ValidateInternalRSASUMTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "FillX"],\
                       [[4095], folds, incs, [1.0, -1.0],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"]])

check_suite.add_checks([checks.ValidateInternalRDZNRM2Test(),\
                        checks.ValidateInternalRDZASUMTest(),\
                        checks.ValidateInternalRSCNRM2Test(),\
                        checks.ValidateInternalRSCASUMTest(),\
                        ],\
                       ["N", "fold", "incX", ("RealScaleX", "ImagScaleX"), "FillX"],\
                       [[4095], folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"]])


check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                        checks.ValidateInternalRSDOTTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "RealScaleY", "FillX", "FillY"],\
                       [[4095], folds, incs, [1.0, -1.0], [1.0, -1.0],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"]])

check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                        checks.ValidateInternalRSDOTTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "RealScaleY", ("FillX", "FillY")],\
                       [[4095], folds, incs, [1.0, -1.0], [1.0, -1.0],\
                        [("constant", "sine"),\
                         ("sine", "constant")]])

check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                        checks.ValidateInternalRSDOTTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "RealScaleY", ("FillX", "FillY")],\
                       [[4095], folds, incs, [1.0, -1.0], [1.0, -1.0],\
                        [("constant", "mountain"),\
                         ("mountain", "constant")]])

check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                        checks.ValidateInternalRZDOTCTest(),\
                        checks.ValidateInternalRCDOTUTest(),\
                        checks.ValidateInternalRCDOTCTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "RealScaleY", "ImagScaleY", "FillX", "FillY"],\
                       [[4095], folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"],\
                        ["constant",\
                         "+big",\
                         "++big",\
                         "+-big"]])

check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                        checks.ValidateInternalRZDOTCTest(),\
                        checks.ValidateInternalRCDOTUTest(),\
                        checks.ValidateInternalRCDOTCTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "RealScaleY", "ImagScaleY", ("FillX", "FillY")],\
                       [[4095], folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        [("constant", "sine"),\
                         ("sine", "constant")]])

check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                        checks.ValidateInternalRZDOTCTest(),\
                        checks.ValidateInternalRCDOTUTest(),\
                        checks.ValidateInternalRCDOTCTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "RealScaleY", "ImagScaleY", ("FillX", "FillY")],\
                       [[4095], folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        [("constant", "mountain"),\
                         ("mountain", "constant")]])

check_suite.add_checks([checks.ValidateInternalRDSUMTest(),\
                        checks.ValidateInternalRDASUMTest(),\
                        checks.ValidateInternalRDNRM2Test(),\
                        checks.ValidateInternalDBDBADDTest(),\
                        checks.ValidateInternalDIDADDTest(),\
                        checks.ValidateInternalDIDDEPOSITTest(),\
                        checks.ValidateInternalRSSUMTest(),\
                        checks.ValidateInternalRSASUMTest(),\
                        checks.ValidateInternalRSNRM2Test(),\
                        checks.ValidateInternalSBSBADDTest(),\
                        checks.ValidateInternalSISADDTest(),\
                        checks.ValidateInternalSISDEPOSITTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "FillX"],\
                       [[255], inf_folds, incs, [1.0, -1.0],\
                        ["+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"]])

check_suite.add_checks([checks.ValidateInternalRZSUMTest(),\
                        checks.ValidateInternalRDZASUMTest(),\
                        checks.ValidateInternalRDZNRM2Test(),\
                        checks.ValidateInternalZBZBADDTest(),\
                        checks.ValidateInternalZIZADDTest(),\
                        checks.ValidateInternalZIZDEPOSITTest(),\
                        checks.ValidateInternalRCSUMTest(),\
                        checks.ValidateInternalRSCASUMTest(),\
                        checks.ValidateInternalRSCNRM2Test(),\
                        checks.ValidateInternalCBCBADDTest(),\
                        checks.ValidateInternalCICADDTest(),\
                        checks.ValidateInternalCICDEPOSITTest()],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX"],\
                       [[255], inf_folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],\
                        ["+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"]])

check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                        checks.ValidateInternalRSDOTTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "RealScaleY", "FillX", "FillY"],\
                       [[255], inf_folds, incs, [1.0, -1.0], [1.0, -1.0],\
                        ["constant",\
                         "+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"],\
                        ["constant",\
                         "+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"]])

check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                        checks.ValidateInternalRZDOTCTest(),\
                        checks.ValidateInternalRCDOTUTest(),\
                        checks.ValidateInternalRCDOTCTest(),\
                        ],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "RealScaleY", "ImagScaleY", "FillX", "FillY"],\
                       [[255], inf_folds, incs, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0],
                        ["constant",\
                         "+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"],\
                        ["constant",\
                         "+inf",\
                         "++inf",\
                         "+-inf",\
                         "nan",\
                         "+inf_nan",\
                         "++inf_nan",\
                         "+-inf_nan"]])

check_suite.add_checks([checks.ValidateXBLASRDDOTTest(),\
                        checks.ValidateXBLASRZDOTUTest(),\
                        checks.ValidateXBLASRZDOTCTest(),\
                        checks.ValidateXBLASRSDOTTest(),\
                        checks.ValidateXBLASRCDOTUTest(),\
                        checks.ValidateXBLASRCDOTCTest()],\
                       ["N", "incX", "incY", "norm"],\
                       [[1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 63, 64, 4095, 4096], [1, 2, 4], [1, 2, 4], [-1, 0, 1]])


check_suite.add_checks([checks.VerifyRDSUMTest(),\
                        checks.VerifyRDASUMTest(),\
                        checks.VerifyDBDBADDTest(),\
                        checks.VerifyDIDADDTest(),\
                        checks.VerifyDIDDEPOSITTest(),\
                        checks.VerifyRZSUMTest(),\
                        checks.VerifyRDZASUMTest(),\
                        checks.VerifyZBZBADDTest(),\
                        checks.VerifyZIZADDTest(),\
                        checks.VerifyZIZDEPOSITTest(),\
                        checks.VerifyRSSUMTest(),\
                        checks.VerifyRSASUMTest(),\
                        checks.VerifySBSBADDTest(),\
                        checks.VerifySISADDTest(),\
                        checks.VerifySISDEPOSITTest(),\
                        checks.VerifyRCSUMTest(),\
                        checks.VerifyRSCASUMTest(),\
                        checks.VerifyCBCBADDTest(),\
                        checks.VerifyCICADDTest(),\
                        checks.VerifyCICDEPOSITTest()],\
                       ["N", "fold", "B", "incX", "RealScaleX", "FillX"],\
                       [[4095], folds, [256], incs, [0],\
                        ["constant"]])

check_suite.add_checks([checks.VerifyRDSUMTest(),\
                        checks.VerifyRDASUMTest(),\
                        checks.VerifyRDNRM2Test(),\
                        checks.VerifyDIDSSQTest(),\
                        checks.VerifyDBDBADDTest(),\
                        checks.VerifyDIDADDTest(),\
                        checks.VerifyDIDDEPOSITTest(),\
                        checks.VerifyRZSUMTest(),\
                        checks.VerifyRDZASUMTest(),\
                        checks.VerifyRDZNRM2Test(),\
                        checks.VerifyDIZSSQTest(),\
                        checks.VerifyZBZBADDTest(),\
                        checks.VerifyZIZADDTest(),\
                        checks.VerifyZIZDEPOSITTest(),\
                        checks.VerifyRSSUMTest(),\
                        checks.VerifyRSASUMTest(),\
                        checks.VerifyRSNRM2Test(),\
                        checks.VerifySISSSQTest(),\
                        checks.VerifySBSBADDTest(),\
                        checks.VerifySISADDTest(),\
                        checks.VerifySISDEPOSITTest(),\
                        checks.VerifyRCSUMTest(),\
                        checks.VerifyRSCASUMTest(),\
                        checks.VerifyRSCNRM2Test(),\
                        checks.VerifySICSSQTest(),\
                        checks.VerifyCBCBADDTest(),\
                        checks.VerifyCICADDTest(),\
                        checks.VerifyCICDEPOSITTest()],\
                       ["N", "fold", "B", "incX", "FillX"],\
                       [[4095], folds, [256], incs,\
                        ["rand",\
                         "rand+(rand-1)",\
                         "sine",\
                         "small+grow*big"]])

check_suite.add_checks([checks.VerifyRDDOTTest(),\
                        checks.VerifyRZDOTUTest(),\
                        checks.VerifyRZDOTCTest(),\
                        checks.VerifyRSDOTTest(),\
                        checks.VerifyRCDOTUTest(),\
                        checks.VerifyRCDOTCTest()],\
                       ["N", "fold", "incX", "incY", "FillX", "FillY"],\
                       [[4095], folds, incs, incs,\
                        ["rand",\
                         "rand+(rand-1)",\
                         "sine",\
                         "small+grow*big"],\
                        ["rand",\
                         "rand+(rand-1)",\
                         "sine",\
                         "small+grow*big"]])


for i in range(DBL_BIN_WIDTH + 2):
  check_suite.add_checks([checks.ValidateInternalRDSUMTest(),\
                          checks.ValidateInternalDBDBADDTest(),\
                          checks.ValidateInternalDIDADDTest(),\
                          checks.ValidateInternalDIDDEPOSITTest(),\
                          checks.ValidateInternalRDASUMTest(),\
                          checks.ValidateInternalRDNRM2Test(),\
                          checks.ValidateInternalRDDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[8192], folds, incs, [DBL_ONES + 2 ** i],\
                          ["constant"],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRDSUMTest(),\
                          checks.ValidateInternalDBDBADDTest(),\
                          checks.ValidateInternalDIDADDTest(),\
                          checks.ValidateInternalDIDDEPOSITTest(),\
                          checks.ValidateInternalRDASUMTest(),\
                          checks.ValidateInternalRDNRM2Test(),\
                          checks.ValidateInternalRDDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          ["constant"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRDDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(DBL_MAX_EXP - 2 * DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - 2 * DBL_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRZSUMTest(),\
                          checks.ValidateInternalZBZBADDTest(),\
                          checks.ValidateInternalZIZADDTest(),\
                          checks.ValidateInternalZIZDEPOSITTest(),\
                          checks.ValidateInternalRDZASUMTest(),\
                          checks.ValidateInternalRDZNRM2Test(),\
                          checks.ValidateInternalRZDOTUTest(),\
                          checks.ValidateInternalRZDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                          checks.ValidateInternalRZDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          [1.5 * 2**(DBL_MAX_EXP - DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - DBL_SMALL_EXP + i)],\
                          ["constant"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRZDOTUTest(),\
                          checks.ValidateInternalRZDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(DBL_MAX_EXP - 2 * DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - 2 * DBL_SMALL_EXP + i)],\
                          [1.5 * 2**(DBL_MAX_EXP - 2 * DBL_BIG_EXP - 6 - i), 0.75 * 2**(DBL_MIN_EXP - 2 * DBL_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

for i in range(FLT_BIN_WIDTH + 2):
  check_suite.add_checks([checks.ValidateInternalRSSUMTest(),\
                          checks.ValidateInternalSBSBADDTest(),\
                          checks.ValidateInternalSISADDTest(),\
                          checks.ValidateInternalSISDEPOSITTest(),\
                          checks.ValidateInternalRSASUMTest(),\
                          checks.ValidateInternalRSNRM2Test(),\
                          checks.ValidateInternalRSDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[8192], folds, incs, [FLT_ONES * 2.0 ** i],\
                          ["constant",],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRSSUMTest(),\
                          checks.ValidateInternalSBSBADDTest(),\
                          checks.ValidateInternalSISADDTest(),\
                          checks.ValidateInternalSISDEPOSITTest(),\
                          checks.ValidateInternalRSASUMTest(),\
                          checks.ValidateInternalRSNRM2Test(),\
                          checks.ValidateInternalRSDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRSDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          ["constant"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRSDOTTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "FillX", "FillY"],\
                         [[32], folds, incs,\
                         [1.5 * 2**(FLT_MAX_EXP - 2 * FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - 2 * FLT_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRCSUMTest(),\
                          checks.ValidateInternalCBCBADDTest(),\
                          checks.ValidateInternalCICADDTest(),\
                          checks.ValidateInternalCICDEPOSITTest(),\
                          checks.ValidateInternalRSCASUMTest(),\
                          checks.ValidateInternalRSCNRM2Test(),\
                          checks.ValidateInternalRCDOTUTest(),\
                          checks.ValidateInternalRCDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant"]])

  check_suite.add_checks([checks.ValidateInternalRCDOTUTest(),\
                          checks.ValidateInternalRCDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          [1.5 * 2**(FLT_MAX_EXP - FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - FLT_SMALL_EXP + i)],\
                          ["constant"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

  check_suite.add_checks([checks.ValidateInternalRCDOTUTest(),\
                          checks.ValidateInternalRCDOTCTest(),\
                          ],\
                         ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX", "FillY"],\
                         [[16], folds, incs,\
                          [1.5 * 2**(FLT_MAX_EXP - 2 * FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - 2 * FLT_SMALL_EXP + i)],\
                          [1.5 * 2**(FLT_MAX_EXP - 2 * FLT_BIG_EXP - 6 - i), 0.75 * 2**(FLT_MIN_EXP - 2 * FLT_SMALL_EXP + i)],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"],\
                          ["constant",\
                           "+big",\
                           "++big",\
                           "+-big"]])

check_suite.add_checks([checks.ValidateInternalDBDBADDTest(),\
                        checks.ValidateInternalDIDADDTest(),\
                        checks.ValidateInternalDIDDEPOSITTest(),\
                        checks.ValidateInternalRDSUMTest(),\
                        checks.ValidateInternalRDASUMTest(),\
                        checks.ValidateInternalRDDOTTest(),\
                        checks.ValidateInternalZBZBADDTest(),\
                        checks.ValidateInternalZIZADDTest(),\
                        checks.ValidateInternalZIZDEPOSITTest(),\
                        checks.ValidateInternalRZSUMTest(),\
                        checks.ValidateInternalRDZASUMTest(),\
                        checks.ValidateInternalRZDOTUTest(),\
                        checks.ValidateInternalRZDOTCTest()],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX"],\
                       [[1], folds, incs, [DBL_ONES * 2 **(DBL_MAX_EXP - 1), 1.0], [DBL_ONES * 2 **(DBL_MAX_EXP - 1), 1.0],\
                        ["constant",]])

check_suite.add_checks([checks.ValidateInternalSBSBADDTest(),\
                        checks.ValidateInternalSISADDTest(),\
                        checks.ValidateInternalSISDEPOSITTest(),\
                        checks.ValidateInternalRSSUMTest(),\
                        checks.ValidateInternalRSASUMTest(),\
                        checks.ValidateInternalRSDOTTest(),\
                        checks.ValidateInternalCBCBADDTest(),\
                        checks.ValidateInternalCICADDTest(),\
                        checks.ValidateInternalCICDEPOSITTest(),\
                        checks.ValidateInternalRCSUMTest(),\
                        checks.ValidateInternalRSCASUMTest(),\
                        checks.ValidateInternalRCDOTUTest(),\
                        checks.ValidateInternalRCDOTCTest()],\
                       ["N", "fold", "incX", "RealScaleX", "ImagScaleX", "FillX"],\
                       [[1], folds, incs, [FLT_ONES * 2 **(FLT_MAX_EXP - 1), 1.0], [FLT_ONES * 2 **(FLT_MAX_EXP - 1), 1.0],\
                        ["constant",]])

check_suite.add_checks([checks.CorroborateRDGEMVTest(),\
                        checks.CorroborateRZGEMVTest(),\
                        checks.CorroborateRSGEMVTest(),\
                        checks.CorroborateRCGEMVTest(),\
                        ],\
                       ["O", "T", "M", "N", "lda", ("incX", "incY"), "FillA", "FillX", "FillY", ("RealAlpha", "ImagAlpha"), ("RealBeta", "ImagBeta"), "fold"],\
                       [["RowMajor", "ColMajor"], ["Trans", "NoTrans"], [255, 512], [255, 512], [0, -15], list(zip(incs, incs)),\
                        ["rand",\
                         ],\
                        ["rand",\
                         ],\
                        ["rand"],\
                        [(0.0, 0.0), (1.0, 0.0), (2.0, 2.0)],\
                        [(0.0, 0.0), (1.0, 0.0), (2.0, 2.0)],\
                        folds])


check_suite.add_checks([checks.CorroborateRDGEMMTest(),
                        checks.CorroborateRZGEMMTest(),\
                        checks.CorroborateRSGEMMTest(),\
                        checks.CorroborateRCGEMMTest(),\
                        ],\
                       ["O", "TransA", "TransB", "M", "N", "K", ("lda", "ldb", "ldc"), "FillA", "FillB", "FillC", ("RealAlpha", "ImagAlpha"), ("RealBeta", "ImagBeta"), "fold"],\
                       [["RowMajor", "ColMajor"], ["ConjTrans", "Trans", "NoTrans"], ["ConjTrans", "Trans", "NoTrans"], [32, 64], [32, 64], [32, 64], [(0, 0, 0), (-63, -63, -63)], \
                        ["rand",\
                         ],\
                        ["rand",\
                         ],\
                        ["rand"],\
                        [(0.0, 0.0), (1.0, 0.0), (2.0, 2.0)],\
                        [(0.0, 0.0), (1.0, 0.0), (2.0, 2.0)],\
                        folds])

check_harness = harness.Harness("check")
check_harness.add_suite(check_suite)
check_harness.run()
