===============旋转阵列=================
        # # 径向
        # for i in range(self.n_r):
        #     elset = self.elset["surface_cable"]["radial"][i]
        #     self.rotate_mesh(elset)
        # # 环向
        # for i in range(self.n_r-1):
        #     elset = self.elset["surface_cable"]["circumferential"][i]
        #     self.rotate_mesh(elset)
        # # 张拉
        # for i in range(self.n_r-1):
        #     elset = self.elset["surface_cable"]["tie"][i]
        #     self.rotate_mesh(elset)
        # # 支撑索
        # for elset in self.elset["support_cable"].values():
        #     self.rotate_mesh(elset[0])
        # # # 3. 类似 HyperMesh 的共节点处理
        # self.unif