OPENQASM 2.0;
include "qelib1.inc";

qreg q[20];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
cx q[1],q[0];
rz(4.97358886312869e-03*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(4.97358886312869e-03*pi) q[1];
cx q[2],q[1];
s q[1];
cx q[3],q[2];
h q[1];
rz(4.97358886312869e-03*pi) q[2];
s q[1];
cx q[3],q[2];
s q[2];
cx q[4],q[3];
h q[2];
rz(4.97358886312869e-03*pi) q[3];
s q[2];
cx q[4],q[3];
s q[3];
cx q[5],q[4];
h q[3];
rz(4.97358886312869e-03*pi) q[4];
s q[3];
cx q[5],q[4];
s q[4];
cx q[6],q[5];
h q[4];
rz(4.97358886312869e-03*pi) q[5];
s q[4];
cx q[6],q[5];
s q[5];
cx q[7],q[6];
h q[5];
rz(4.97358886312869e-03*pi) q[6];
s q[5];
cx q[7],q[6];
s q[6];
cx q[8],q[7];
h q[6];
rz(4.97358886312869e-03*pi) q[7];
s q[6];
cx q[8],q[7];
s q[7];
cx q[9],q[8];
h q[7];
rz(4.97358886312869e-03*pi) q[8];
s q[7];
cx q[9],q[8];
s q[8];
cx q[10],q[9];
h q[8];
rz(4.97358886312869e-03*pi) q[9];
s q[8];
cx q[10],q[9];
s q[9];
cx q[11],q[10];
h q[9];
rz(4.97358886312869e-03*pi) q[10];
s q[9];
cx q[11],q[10];
s q[10];
cx q[12],q[11];
h q[10];
rz(4.97358886312869e-03*pi) q[11];
s q[10];
cx q[12],q[11];
s q[11];
cx q[13],q[12];
h q[11];
rz(4.97358886312869e-03*pi) q[12];
s q[11];
cx q[13],q[12];
s q[12];
cx q[14],q[13];
h q[12];
rz(4.97358886312869e-03*pi) q[13];
s q[12];
cx q[14],q[13];
s q[13];
cx q[15],q[14];
h q[13];
rz(4.97358886312869e-03*pi) q[14];
s q[13];
cx q[15],q[14];
s q[14];
cx q[16],q[15];
h q[14];
rz(4.97358886312869e-03*pi) q[15];
s q[14];
cx q[16],q[15];
s q[15];
cx q[17],q[16];
h q[15];
rz(4.97358886312869e-03*pi) q[16];
s q[15];
cx q[17],q[16];
s q[16];
cx q[18],q[17];
h q[16];
rz(4.97358886312869e-03*pi) q[17];
s q[16];
cx q[18],q[17];
s q[17];
cx q[19],q[18];
h q[17];
rz(4.97358886312869e-03*pi) q[18];
s q[17];
cx q[19],q[18];
cx q[19],q[0];
s q[18];
rz(4.97358886312869e-03*pi) q[0];
h q[18];
cx q[19],q[0];
s q[18];
s q[0];
s q[19];
h q[0];
h q[19];
s q[0];
s q[19];
cx q[1],q[0];
rz(4.97358886312869e-03*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(4.97358886312869e-03*pi) q[1];
cx q[2],q[1];
h q[1];
cx q[3],q[2];
rz(4.97358886312869e-03*pi) q[2];
cx q[3],q[2];
h q[2];
cx q[4],q[3];
rz(4.97358886312869e-03*pi) q[3];
cx q[4],q[3];
h q[3];
cx q[5],q[4];
rz(4.97358886312869e-03*pi) q[4];
cx q[5],q[4];
h q[4];
cx q[6],q[5];
rz(4.97358886312869e-03*pi) q[5];
cx q[6],q[5];
h q[5];
cx q[7],q[6];
rz(4.97358886312869e-03*pi) q[6];
cx q[7],q[6];
h q[6];
cx q[8],q[7];
rz(4.97358886312869e-03*pi) q[7];
cx q[8],q[7];
h q[7];
cx q[9],q[8];
rz(4.97358886312869e-03*pi) q[8];
cx q[9],q[8];
h q[8];
cx q[10],q[9];
rz(4.97358886312869e-03*pi) q[9];
cx q[10],q[9];
h q[9];
cx q[11],q[10];
rz(4.97358886312869e-03*pi) q[10];
cx q[11],q[10];
h q[10];
cx q[12],q[11];
rz(4.97358886312869e-03*pi) q[11];
cx q[12],q[11];
h q[11];
cx q[13],q[12];
rz(4.97358886312869e-03*pi) q[12];
cx q[13],q[12];
h q[12];
cx q[14],q[13];
rz(4.97358886312869e-03*pi) q[13];
cx q[14],q[13];
h q[13];
cx q[15],q[14];
rz(4.97358886312869e-03*pi) q[14];
cx q[15],q[14];
h q[14];
cx q[16],q[15];
rz(4.97358886312869e-03*pi) q[15];
cx q[16],q[15];
h q[15];
cx q[17],q[16];
rz(4.97358886312869e-03*pi) q[16];
cx q[17],q[16];
h q[16];
cx q[18],q[17];
rz(4.97358886312869e-03*pi) q[17];
cx q[18],q[17];
h q[17];
cx q[19],q[18];
rz(4.97358886312869e-03*pi) q[18];
cx q[19],q[18];
cx q[19],q[0];
h q[18];
rz(4.97358886312869e-03*pi) q[0];
cx q[19],q[0];
h q[0];
h q[19];
cx q[1],q[0];
rz(4.97358886312869e-03*pi) q[0];
cx q[1],q[0];
cx q[2],q[1];
rz(4.97358886312869e-03*pi) q[1];
cx q[2],q[1];
rz(0.5000000044757823*pi) q[1];
cx q[3],q[2];
rz(4.97358886312869e-03*pi) q[2];
cx q[3],q[2];
rz(0.5000000030765888*pi) q[2];
cx q[4],q[3];
rz(4.97358886312869e-03*pi) q[3];
cx q[4],q[3];
rz(0.5000000010386961*pi) q[3];
cx q[5],q[4];
rz(4.97358886312869e-03*pi) q[4];
cx q[5],q[4];
rz(0.5000000045802251*pi) q[4];
cx q[6],q[5];
rz(4.97358886312869e-03*pi) q[5];
cx q[6],q[5];
rz(0.5000000002706845*pi) q[5];
cx q[7],q[6];
rz(4.97358886312869e-03*pi) q[6];
cx q[7],q[6];
rz(0.5000000044630106*pi) q[6];
cx q[8],q[7];
rz(4.97358886312869e-03*pi) q[7];
cx q[8],q[7];
rz(0.5000000025485786*pi) q[7];
cx q[9],q[8];
rz(4.97358886312869e-03*pi) q[8];
cx q[9],q[8];
rz(0.5000000012106233*pi) q[8];
cx q[10],q[9];
rz(4.97358886312869e-03*pi) q[9];
cx q[10],q[9];
rz(0.5000000020880128*pi) q[9];
cx q[11],q[10];
rz(4.97358886312869e-03*pi) q[10];
cx q[11],q[10];
rz(0.5000000035573443*pi) q[10];
cx q[12],q[11];
rz(4.97358886312869e-03*pi) q[11];
cx q[12],q[11];
rz(0.5000000022775827*pi) q[11];
cx q[13],q[12];
rz(4.97358886312869e-03*pi) q[12];
cx q[13],q[12];
rz(0.5000000049675641*pi) q[12];
cx q[14],q[13];
rz(4.97358886312869e-03*pi) q[13];
cx q[14],q[13];
rz(0.5000000045572774*pi) q[13];
cx q[15],q[14];
rz(4.97358886312869e-03*pi) q[14];
cx q[15],q[14];
rz(0.5000000032078445*pi) q[14];
cx q[16],q[15];
rz(4.97358886312869e-03*pi) q[15];
cx q[16],q[15];
rz(0.5000000031441776*pi) q[15];
cx q[17],q[16];
rz(4.97358886312869e-03*pi) q[16];
cx q[17],q[16];
rz(0.5000000021537957*pi) q[16];
cx q[18],q[17];
rz(4.97358886312869e-03*pi) q[17];
cx q[18],q[17];
rz(0.5000000042055095*pi) q[17];
cx q[19],q[18];
rz(4.97358886312869e-03*pi) q[18];
cx q[19],q[18];
cx q[19],q[0];
rz(0.5000000027338896*pi) q[18];
rz(4.97358886312869e-03*pi) q[0];
cx q[19],q[0];
rz(0.5000000003868974*pi) q[0];
rz(0.5000000012778284*pi) q[19];