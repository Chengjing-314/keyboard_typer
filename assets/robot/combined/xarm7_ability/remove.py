f = open(
    "/home/chengjing/Desktop/bimanual/assets/robot/heti/xarm7_ability/xarm7_ability_left_hand_glb.urdf"
)
lines = f.readlines()
f1 = open(
    "/home/chengjing/Desktop/bimanual/assets/robot/heti/xarm7_ability/xarm7_ability_left_hand_glb_nmm_cs.urdf",
    "w",
)
for line in lines:
    if "<mimic" in line:
        continue
    # elif "<origin" in line:
    #     f1.write("      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n")
    #     continue
    f1.write(line)
