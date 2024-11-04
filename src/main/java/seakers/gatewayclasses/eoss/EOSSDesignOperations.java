package seakers.gatewayclasses.eoss;

import com.mathworks.engine.MatlabEngine;
import org.moeaframework.core.Variation;
import org.moeaframework.problem.AbstractProblem;
import py4j.GatewayServer;
import seakers.trussaos.architecture.TrussRepeatableArchitecture;

import java.util.ArrayList;

public class EOSSDesignOperations {

    // Parameters for problem class
    private String savePath;
    private int modelSelection;
    private int numberOfVariables; // computed depending on the size of the node grid considered (3x3, 5x5 etc.)
    private int numberOfHeuristicObjectives;
    private int numberOfHeuristicConstraints;
    private String[] instrumentNames;
    private String[] orbitNames;
    private boolean satelliteProblem;
    private AbstractProblem problem;

    // Parameters for RL action and reward computation
    private ArrayList<String> objectiveNames;
    private ArrayList<String> constraintNames;
    private ArrayList<String> heuristicNames;
    private ArrayList<Double> objectives;
    private ArrayList<Double> constraints;
    private ArrayList<Double> heuristics;
    private TrussRepeatableArchitecture currentDesign;
    private int action; // integer representing which design variable (corresponding to the instrument in the satellite) to add or remove
    private ArrayList<Boolean> heuristicsDeployed;
    private ArrayList<Variation> heuristicOperators;
    private int[] newDesign;

    /**
     * Constructor for EOSS operations class instance
     * sets the problem instance, modifies a design based on the action and evaluates a design
     */

    public EOSSDesignOperations() {

    }




}
