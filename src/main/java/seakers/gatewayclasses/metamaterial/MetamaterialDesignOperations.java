package seakers.gatewayclasses.metamaterial;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.problem.AbstractProblem;
import seakers.trussaos.architecture.TrussRepeatableArchitecture;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;
import java.util.ArrayList;

public class MetamaterialDesignOperations {

    // Parameters for problem class
    private String savePath;
    private MatlabEngine engine;
    private int modelSelection;
    private int numberOfVariables; // computed depending on the size of the node grid considered (3x3, 5x5 etc.)
    private int numberOfHeuristicObjectives;
    private int numberOfHeuristicConstraints;
    private double sideElementLength;
    private double sideNodeNumber;
    private double radius;
    private double YoungsModulus;
    private double targetStiffnessRatio;
    private double nucFac; // nucFac is not used in the current Periodic Boundary Conditions model but is still an argument to the class instance
    private boolean arteryProblem;
    private AbstractProblem problem;

    // Parameters for RL action and reward computation
    private double[][] fullNodalConnectivityArray;
    private ArrayList<String> objectiveNames;
    private ArrayList<String> constraintNames;
    private ArrayList<String> heuristicNames;
    private ArrayList<Double> objectives;
    private ArrayList<Double> constraints;
    private ArrayList<Double> heuristics;
    private TrussRepeatableArchitecture currentDesign;
    private int action;
    private TrussRepeatableArchitecture newDesign;

    /**
     * Constructor for class instance, only starts the Matlab engine
     */
    public MetamaterialDesignOperations() throws InterruptedException, EngineException {
        this.engine = MatlabEngine.startMatlab();
    }

    // Set parameter for problem being solved (true -> Artery, false -> Equal Stiffness)
    public void setArteryProblem(boolean arteryProblem) {
        this.arteryProblem = arteryProblem;
    }

    // Run parameter setting methods before initializing problem instance parameter
    public void setSavePath(String savePath) {
        this.savePath = savePath;
    }

    public void setModelSelection(int selectedModel) {
        this.modelSelection = selectedModel;
    }

    public void setNumberOfVariables(int numberOfVariables) {
        this.numberOfVariables = numberOfVariables;
    }

    public void setSideElementLength(double sel) {
        this.sideElementLength = sel;
    }

    public void setSideNodeNumber(double sidenum) {
        this.sideNodeNumber = sidenum;
    }

    public void setRadius(double rad) {
        this.radius = rad;
    }

    public void setYoungsModulus(double E) {
        this.YoungsModulus = E;
    }

    public void setTargetStiffnessRatio(double targetStiffnessRatio) {
        this.targetStiffnessRatio = targetStiffnessRatio;
    }

    public void setNucFac(double nucFac) {
        this.nucFac = nucFac;
    }

    public void setFullNodalConnectivityArray(double[][] nodalConnectivityArray) {
        this.fullNodalConnectivityArray = nodalConnectivityArray;
    }

    // Run after running setting methods
    public void setProblem() {
        // Heuristic Enforcement Methods
        /**
         * partialCollapsibilityConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * nodalPropertiesConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * orientationConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * intersectionConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         *
         * heuristicsConstrained = [partialCollapsibilityConstrained, nodalPropertiesConstrained, orientationConstrained, intersectionConstrained]
         */
        boolean[] partialCollapsibilityConstrained = {false, false, false, false, false, false, false};
        boolean[] nodalPropertiesConstrained = {false, false, false, false, false, false, false};
        boolean[] orientationConstrained = {false, false, false, false, false, false, false};
        boolean[] intersectionConstrained = {false, false, false, false, false, false, false};

        boolean[][] heuristicsConstrained = new boolean[4][7];
        for (int i = 0; i < 7; i++) {
            heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
            heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
            heuristicsConstrained[2][i] = orientationConstrained[i];
            heuristicsConstrained[3][i] = intersectionConstrained[i];
        }

        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        for (int i = 0; i < 4; i++) {
            if (heuristicsConstrained[i][5]) {
                numberOfHeuristicConstraints++;
            }
            if (heuristicsConstrained[i][4]) {
                numberOfHeuristicObjectives++;
            }
        }

        this.numberOfHeuristicObjectives = numberOfHeuristicObjectives;
        this.numberOfHeuristicConstraints = numberOfHeuristicConstraints;
        if (this.arteryProblem) {
            this.problem = new ConstantRadiusArteryProblem(this.savePath, this.modelSelection, this.numberOfVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, this.radius, this.sideElementLength, this.YoungsModulus, this.sideNodeNumber, this.nucFac, this.targetStiffnessRatio, engine, heuristicsConstrained);
        } else {
            this.problem = new ConstantRadiusTrussProblem2(this.savePath, this.modelSelection, this.numberOfVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, this.radius, this.sideElementLength, this.YoungsModulus, this.sideNodeNumber, this.nucFac, this.targetStiffnessRatio, engine, heuristicsConstrained);
        }

    }

    // Setting methods for data saving
    public void setObjectiveNames(ArrayList<String> objectiveNames) {
        this.objectiveNames = objectiveNames;
    }

    public void setConstraintNames(ArrayList<String> constraintNames) {
        this.constraintNames = constraintNames;
    }

    public void setHeuristicNames(ArrayList<String> heuristicNames) {
        this.heuristicNames = heuristicNames;
    }

    // Reset method for new design evaluation and manipulation
    public void resetDesignGoals() {
        this.objectives = new ArrayList<>();
        this.constraints = new ArrayList<>();
        this.heuristicNames = new ArrayList<>();
        this.newDesign = new TrussRepeatableArchitecture(new Solution(this.problem.getNumberOfVariables(), this.problem.getNumberOfObjectives(), this.problem.getNumberOfConstraints()), this.sideNodeNumber, 0, 0);
    }

    public void setCurrentDesign(ArrayList<Boolean> design) {
        Solution currentSolution = new Solution(this.problem.getNumberOfVariables(), this.problem.getNumberOfObjectives(), this.problem.getNumberOfConstraints());
        for (int i = 0; i < design.size(); i++) {
            BinaryVariable var = new BinaryVariable(1);
            EncodingUtils.setBoolean(var, design.get(i));
            currentSolution.setVariable(i, var);
        }
        this.currentDesign = new TrussRepeatableArchitecture(currentSolution, this.sideNodeNumber, this.numberOfHeuristicObjectives, this.numberOfHeuristicConstraints);
    }

    public ArrayList<Double> evaluate() {
        this.problem.evaluate(this.currentDesign);

        // Results are stored in the order -> {objectives, constraints, heuristics}
        ArrayList<Double> designMetrics = new ArrayList<>();
        ArrayList<Double> currentObjectives = new ArrayList<>();
        for (int i = 0; i < this.objectiveNames.size(); i++) {
            currentObjectives.add(this.currentDesign.getObjective(i));
        }

        ArrayList<Double> currentConstraints = new ArrayList<>();
        for (String name : this.constraintNames) {
            currentConstraints.add((double) this.currentDesign.getAttribute(name));
        }

        ArrayList<Double> currentHeurisics = new ArrayList<>();
        for (String name : this.heuristicNames) {
            currentHeurisics.add((double) this.currentDesign.getAttribute(name));
        }

        designMetrics.addAll(currentObjectives);
        designMetrics.addAll(currentConstraints);
        designMetrics.addAll(currentHeurisics);

        this.objectives = currentObjectives;
        this.constraints = currentConstraints;
        this.heuristics = currentHeurisics;

        return designMetrics;
    }

    public void setAction(int action) {
        this.action = action;
    }

    public void operate() {
        double[] memberToAdd = this.fullNodalConnectivityArray[this.action];

        // If an edge member is to be added, add the opposite member to preserve repeatability
        if (isEdgeMember(memberToAdd)) {
            double[] repeatableMember = getOppositeMember(memberToAdd);
        }

        ///////////// POPULATE ///////////////


    }

    private boolean isEdgeMember(double[] member) {
        boolean isEdge = false;

        ///////////// POPULATE ///////////////

        return isEdge;
    }

    private double[] getOppositeMember(double[] member) {
        double[] oppositeMember = new double[member.length];

        ///////////// POPULATE ///////////////

        return oppositeMember;
    }

    //////// IMPLEMENT IN PYTHON /////////
    private double[][] getCompleteConnectivityArray(){
        int memberCount = 0;
        //int[] nodesArray = IntStream.range(1,sidenum*sidenum).toArray();
        int totalNumberOfMembers;
        if (this.sideNodeNumber >= 5) {
            int sidenumSquared = (int) (this.sideNodeNumber*this.sideNodeNumber);
            totalNumberOfMembers =  sidenumSquared * (sidenumSquared - 1)/2;
        }
        else {
            totalNumberOfMembers = (int) (CombinatoricsUtils.factorial((int) (this.sideNodeNumber*this.sideNodeNumber))/(CombinatoricsUtils.factorial((int) ((this.sideNodeNumber*this.sideNodeNumber) - 2)) * CombinatoricsUtils.factorial(2)));
        }
        double[][] completeConnectivityArray = new double[totalNumberOfMembers][2];
        for (int i = 0; i < ((this.sideNodeNumber*this.sideNodeNumber)-1); i++) {
            for (int j = i+1; j < (this.sideNodeNumber*this.sideNodeNumber); j++) {
                completeConnectivityArray[memberCount][0] = i+1;
                completeConnectivityArray[memberCount][1] = j+1;
                memberCount += 1;
            }
        }
        return completeConnectivityArray;
    }

}
