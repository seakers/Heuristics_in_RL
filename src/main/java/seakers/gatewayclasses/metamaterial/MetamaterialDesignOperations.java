package seakers.gatewayclasses.metamaterial;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variation;
import org.moeaframework.core.operator.CompoundVariation;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.problem.AbstractProblem;
import seakers.trussaos.architecture.TrussRepeatableArchitecture;
import seakers.trussaos.operators.constantradii.AddDiagonalMember;
import seakers.trussaos.operators.constantradii.AddMember;
import seakers.trussaos.operators.constantradii.ImproveOrientation2;
import seakers.trussaos.operators.constantradii.RemoveIntersection2;
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
    private boolean[] heuristicsDeployed;
    private ArrayList<Variation> heuristicOperators;
    private TrussRepeatableArchitecture currentDesign;
    private int action; // integer representing which design variable (corresponding to the repeatable truss) to add or remove
    private boolean[] newDesign;

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

    // Setting method for heuristics deployed
    public void setHeuristicsDeployed(boolean[] heuristicsDeployed) {
        this.heuristicsDeployed = heuristicsDeployed;
    }

    // Run after running setting methods for parameters and heuristics deployed
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

        double[][] globalNodePositions;
        if (this.arteryProblem) {
            this.problem = new ConstantRadiusArteryProblem(this.savePath, this.modelSelection, this.numberOfVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, this.radius, this.sideElementLength, this.YoungsModulus, this.sideNodeNumber, this.nucFac, this.targetStiffnessRatio, engine, heuristicsConstrained);
            globalNodePositions = ((ConstantRadiusArteryProblem) this.problem).getNodalConnectivityArray();
        } else {
            this.problem = new ConstantRadiusTrussProblem2(this.savePath, this.modelSelection, this.numberOfVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, this.radius, this.sideElementLength, this.YoungsModulus, this.sideNodeNumber, this.nucFac, this.targetStiffnessRatio, engine, heuristicsConstrained);
            globalNodePositions = ((ConstantRadiusTrussProblem2) this.problem).getNodalConnectivityArray();
        }

        boolean maintainFeasibility = false;
        double mutationProbability = 1. / this.problem.getNumberOfVariables();

        // Order of heuristic operators -> {partial collapsibility, nodal properties, orientation, intersection}
        Variation addMember = new CompoundVariation(new AddMember(maintainFeasibility, this.arteryProblem, this.engine, globalNodePositions, this.sideNodeNumber, this.sideElementLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
        Variation removeIntersection = new CompoundVariation(new RemoveIntersection2(this.arteryProblem, this.engine, globalNodePositions, this.sideNodeNumber, this.sideElementLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
        Variation addDiagonalMember = new CompoundVariation(new AddDiagonalMember(maintainFeasibility, this.arteryProblem, this.engine, globalNodePositions, this.sideNodeNumber, this.sideElementLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
        Variation improveOrientation = new CompoundVariation(new ImproveOrientation2(this.arteryProblem, globalNodePositions, this.targetStiffnessRatio, (int) this.sideNodeNumber, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));

        Variation[] allHeuristicOperators = new Variation[]{addDiagonalMember, addMember, improveOrientation, removeIntersection};

        ArrayList<Variation> deployedHeuristicOperators = new ArrayList<>();
        for (int j = 0; j < allHeuristicOperators.length; j++) {
            if (this.heuristicsDeployed[j]) {
                deployedHeuristicOperators.add(allHeuristicOperators[j]);
            }
        }

        this.heuristicOperators = deployedHeuristicOperators;
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
        this.newDesign = new boolean[this.problem.getNumberOfVariables()];
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
        boolean[] newDesign;
        if (action < (2*this.currentDesign.getNumberOfVariables() + 1)) { // Simple adding/removing of members or no change
            newDesign = this.currentDesign.getBooleanDesignArray(this.currentDesign);

            if (action < this.currentDesign.getNumberOfVariables()) {
                if (!newDesign[action]) { // Add member if its not present in design
                    newDesign[action] = true;
                }
            } else if (action > this.currentDesign.getNumberOfVariables()) { // Remove member if its present in design
                if (newDesign[action - (this.currentDesign.getNumberOfVariables() + 1)]) {
                    newDesign[action - (this.currentDesign.getNumberOfVariables() + 1)] = false;
                }
            } // if action == this.currentDesign.getNumberOfVariables() -> keep the same design
        } else { // heuristic actions
            Variation selectedHeuristicOperator = this.heuristicOperators.get(action - ((2*this.problem.getNumberOfVariables()) + 1));
            Solution newSolution = selectedHeuristicOperator.evolve(new Solution[]{this.currentDesign})[0];
            newDesign = this.currentDesign.getBooleanDesignArray(newSolution);
        }

        this.newDesign = newDesign;
    }

    // Retrieval Methods
    public boolean[] getNewDesign() {
        return this.newDesign;
    }

    public ArrayList<Double> getObjectives() {
        return this.objectives;
    }

    public ArrayList<Double> getConstraints() {
        return this.constraints;
    }

    public ArrayList<Double> getHeuristics() {
        return this.heuristics;
    }

    // Method to obtain full connectivity array for current boolean design
    public double[][] getFullConnectivityArray() {
        boolean[] currentBooleanDesign = currentDesign.getBooleanDesignArray(this.currentDesign);
        return currentDesign.ConvertToFullConnectivityArray(currentBooleanDesign);
    }

    // Method to obtain full connectivity array for new boolean design (run only after operate() method)
    public double[][] getNewDesignConnectivityArray() {
        return currentDesign.ConvertToFullConnectivityArray(this.newDesign);
    }

    //////// MAY NOT BE REQUIRED /////////
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
