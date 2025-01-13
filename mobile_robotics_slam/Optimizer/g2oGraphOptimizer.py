import numpy as np
import g2o
  
    
class GraphOptimizer(): # type: ignore
    def __init__(self):

        self.optimizer = g2o.SparseOptimizer()
        self.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())  # type: ignore
        self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver) # type: ignore
        self.optimizer.set_algorithm(self.algorithm)
        self.optimizer.set_verbose(False)

        # self.optimizer = g2o.SparseOptimizer()
        # self.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        # self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver)
        # self.optimizer.set_algorithm(self.algorithm)
 
    def optimize(self, max_iterations=20):
        print(f'Optimizing Graph with {len(self.optimizer.vertices())} vertices and {len(self.optimizer.edges())} edges')
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(max_iterations)
        self.optimizer.save("out.g2o")

    def fix_pose_vertex_2D(self, id):
        """
        Fix the pose vertex with the given id
        
        Inputs:
            id: Id of the vertex (Int)
        """
        if self.optimizer.vertex(id) is None:
            raise ValueError("Vertex with this id does not exist")
        if type(self.optimizer.vertex(id)) != g2o.VertexSE2:
            raise ValueError("Vertex is not of type VertexSE2")
        self.optimizer.vertex(id).set_fixed(True)




    def add_pose_vertex_2D(self, id, pose, fixed=False):
        """
        Add a pose vertex to the graph with the given id and pose
        
        Inputs:
            id: Id of the vertex (Int)
            pose: Pose of the vertex [x, y, theta] (List)
            fixed: Whether the vertex is fixed or not (Bool)
        """
        #Check if the id is already present
        if self.optimizer.vertex(id) is not None:
            raise ValueError("Vertex with this id already exists")
        #Check if the pose is of the correct dimension
        if len(pose) != 3:
            raise ValueError("Pose should be of the form [x, y, theta]")
        SE2_pose = g2o.SE2(pose[0], pose[1], pose[2]) # type: ignore
        VertexSE2 = g2o.VertexSE2() # type: ignore
        VertexSE2.set_id(id)
        VertexSE2.set_estimate(SE2_pose)
        VertexSE2.set_fixed(fixed)
        self.optimizer.add_vertex(VertexSE2)
        print("Added vertex with id:", id, "And pose:", self.optimizer.vertex(id).estimate().vector())

    def add_landmark_vertex_2D(self, id, position, fixed=False):
        """
        Add a landmark vertex to the graph with the given id and position
        
        Inputs:
            id: Id of the vertex (Int)
            position: Position of the vertex [x, y] (List)
            fixed: Whether the vertex is fixed or not (Bool)
        """
        #Check if the id is already present
        if self.optimizer.vertex(id) is not None:
            raise ValueError("Vertex with this id already exists")
        #Check if the position is of the correct dimension
        if len(position) != 2:
            raise ValueError("Position should be of the form [x, y]")
        
        VertexPointXY = g2o.VertexPointXY() # type: ignore
        VertexPointXY.set_id(id)
        VertexPointXY.set_estimate_data(position)
        VertexPointXY.set_fixed(fixed)
        self.optimizer.add_vertex(VertexPointXY)
        #print("Added vertex with id:", id, "And position:", self.optimizer.vertex(id).estimate())

    def add_odometry_edge_2D(self, Vertex_id1, Vertex_id2, information, robust_kernel=None):
        """
        Add an odometry edge between two pose vertices in the graph with the given information matrix

        Inputs: 
            Vertex_id1: Id of the first vertex (Int)
            Vertex_id2: Id of the second vertex (Int)
            information: Information matrix for the edge (3x3 numpy array)
            robust_kernel: Robust kernel for the edge
        """
        if self.optimizer.vertex(Vertex_id1) is None or self.optimizer.vertex(Vertex_id2) is None:
            raise ValueError("One of the vertices does not exist")
        if type(self.optimizer.vertex(Vertex_id1)) != g2o.VertexSE2 or type(self.optimizer.vertex(Vertex_id2)) != g2o.VertexSE2: # type: ignore
            raise ValueError("One of the vertices is not of type VertexSE2")
        
        kernel = g2o.RobustKernelHuber() # type: ignore
        kernel.set_delta(1.0)  # Set the delta value for the Huber kernel

        edge = g2o.EdgeSE2() # type: ignore
        edge.set_vertex(0, self.optimizer.vertex(Vertex_id1))
        edge.set_vertex(1, self.optimizer.vertex(Vertex_id2))
        edge.set_measurement_from_state()  # relative pose
        edge.set_information(information)
        
        edge.set_robust_kernel(robust_kernel)
        self.optimizer.add_edge(edge)

    def add_pose_landmark_edge_from_state_2D(self, pose_id, landmark_id, information, robust_kernel=None):
        """
        Add an edge between a pose vertex and a landmark vertex in the graph with the given relative measurement and information matrix

        Inputs: 
            pose_id: Id of the pose vertex (Int)
            landmark_id: Id of the landmark vertex (Int)
            relative_measurement: Relative measurement between the two vertices [dx, dy]
            information: Information matrix for the edge (2x2 numpy array)
            robust_kernel: Robust kernel for the edge
        """
        if self.optimizer.vertex(pose_id) is None or self.optimizer.vertex(landmark_id) is None:
            raise ValueError("One of the vertices does not exist")
        if type(self.optimizer.vertex(pose_id)) != g2o.VertexSE2 or type(self.optimizer.vertex(landmark_id)) != g2o.VertexPointXY: # type: ignore
            raise ValueError("One of the vertices is not of the correct type")
        
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1.0)  # Set the delta value for the Huber kernel
        
        edge = g2o.EdgeSE2PointXY() # type: ignore
        edge.set_vertex(0, self.optimizer.vertex(pose_id))
        edge.set_vertex(1, self.optimizer.vertex(landmark_id))
        edge.set_measurement_from_state()  # relative measurement
        edge.set_information(information)
        edge.set_robust_kernel(kernel)
        self.optimizer.add_edge(edge)

    def add_pose_landmark_edge_2D(self, pose_id, landmark_id, relative_measurement,  information, robust_kernel=None):
        """
        Add an edge between a pose vertex and a landmark vertex in the graph with the given relative measurement and information matrix

        Inputs: 
            pose_id: Id of the pose vertex (Int)
            landmark_id: Id of the landmark vertex (Int)
            relative_measurement: Relative measurement between the two vertices [dx, dy]
            information: Information matrix for the edge (2x2 numpy array)
            robust_kernel: Robust kernel for the edge
        """
        if self.optimizer.vertex(pose_id) is None or self.optimizer.vertex(landmark_id) is None:
            raise ValueError("One of the vertices does not exist")
        if type(self.optimizer.vertex(pose_id)) != g2o.VertexSE2 or type(self.optimizer.vertex(landmark_id)) != g2o.VertexPointXY: # type: ignore
            raise ValueError("One of the vertices is not of the correct type")
        
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1.0)  

        edge = g2o.EdgeSE2PointXY() # type: ignore
        edge.set_vertex(0, self.optimizer.vertex(pose_id))
        edge.set_vertex(1, self.optimizer.vertex(landmark_id))
        edge.set_measurement(relative_measurement)
        edge.set_information(information)
        
        edge.set_robust_kernel(kernel)
        self.optimizer.add_edge(edge)


    def add_edge_between_poses_2D(self, Vertex_id1, Vertex_id2, relative_transform, information, robust_kernel=None):
        """
        Add an edge between two pose vertices  in the graph with the given relative transform and information matrix

        Inputs: 
            Vertex_id1: Id of the first vertex (Int)
            Vertex_id2: Id of the second vertex (Int)
            relative_transform: Relative transform between the two vertices [dx, dy, dtheta]
            information: Information matrix for the edge (3x3 numpy array)
            robust_kernel: Robust kernel for the edge
        """
        if self.optimizer.vertex(Vertex_id1) is None or self.optimizer.vertex(Vertex_id2) is None:
            raise ValueError("One of the vertices does not exist")
        if type(self.optimizer.vertex(Vertex_id1)) != g2o.VertexSE2 or type(self.optimizer.vertex(Vertex_id2)) != g2o.VertexSE2: # type: ignore
            raise ValueError("One of the vertices is not of type VertexSE2")
        edge = g2o.EdgeSE2() # type: ignore
        edge.set_vertex(0, self.optimizer.vertex(Vertex_id1)) # 0 = From Vertex
        edge.set_vertex(1, self.optimizer.vertex(Vertex_id2)) # 1 = To Vertex
        edge.set_measurement(g2o.SE2(relative_transform)) # type: ignore
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        self.optimizer.add_edge(edge)

 
    def get_pose_2D(self, id):
        """
        Get the pose of the vertex with the given id
        
        Inputs:
            id: Id of the vertex (Int)
        
        Returns:
            pose: Pose of the vertex [x, y, theta] (List)
        """
        if self.optimizer.vertex(id) is None:
            raise ValueError("Vertex with this id does not exist")
        if type(self.optimizer.vertex(id)) != g2o.VertexSE2: # type: ignore
            raise ValueError("Vertex is not of type VertexSE2")
        return self.optimizer.vertex(id).estimate().vector()
    
    def get_landmark_2D(self, id):
        """
        Get the position of the vertex with the given id
        
        Inputs:
            id: Id of the vertex (Int)
        
        Returns:
            position: Position of the vertex [x, y] (List)
        """
        if self.optimizer.vertex(id) is None:
            raise ValueError("Vertex with this id does not exist")
        if type(self.optimizer.vertex(id)) != g2o.VertexPointXY: # type: ignore
            raise ValueError("Vertex is not of type VertexPointXY")
        return self.optimizer.vertex(id).estimate()
    
    def get_all_poses_2D(self):
        """
        Get the poses of all the pose vertices in the graph sorted by id
        
        Returns:
            poses: Poses of all the pose vertices [[x1, y1, theta1], [x2, y2, theta2], ...] (List of Lists)
        """
        poses = []
        vertices_ids = sorted([v for v in self.optimizer.vertices()])
        for i in vertices_ids:
            if type(self.optimizer.vertex(i)) == g2o.VertexSE2: # type: ignore
                poses.append(self.optimizer.vertex(i).estimate().vector())
        return poses
        
    
    def get_all_landmarks_2D(self):
        """
        Get the positions of all the landmark vertices in the graph
        
        Returns:
            positions: Positions of all the landmark vertices [[x1, y1], [x2, y2], ...] (List of Lists)
        """
        positions = []
        for i in self.optimizer.vertices():
            if type(self.optimizer.vertex(i)) == g2o.VertexPointXY: # type: ignore
                positions.append(self.optimizer.vertex(i).estimate())
        return positions
 
 

