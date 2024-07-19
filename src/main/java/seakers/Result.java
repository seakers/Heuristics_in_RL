package seakers;

import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import seakers.architecture.util.IntegerVariable;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningArchitecture;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringJoiner;

public class Result {
    private final String saveDirectory;

    public Result(String saveDirectory) {
        this.saveDirectory = saveDirectory;
    }

    public void saveAllInternalSolutions(String filename, ArrayList<Solution> solutionSet, String[] variableNames, String[] objectiveNames, String[] constraintNames, String[] heuristicNames, boolean isPartitioning, boolean isEOSS) {

        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving solutions");

        try(FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            headings.add("NFE");
            for (String variableName : variableNames) {
                headings.add(variableName);
            }
            for (String objectiveName : objectiveNames) {
                headings.add(objectiveName);
            }
            for (String constraintName : constraintNames) {
                headings.add(constraintName);
            }
            for (String heuristicName : heuristicNames) {
                headings.add(heuristicName);
            }
            writer.append(headings.toString());
            writer.append("\n");

            for (Solution solution : solutionSet) {
                StringJoiner sj = new StringJoiner(",");
                sj.add(Integer.toString((Integer) solution.getAttribute("NFE")));
                if (isPartitioning) { // Only for the Partitioning problem, extract integer variables and store
                    PartitioningArchitecture arch = (PartitioningArchitecture) solution;
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Integer.toString(((IntegerVariable)arch.getVariable(i)).getValue()));
                    }
                } else { // Other problems don't use integer variables and Assigning problem does not require the first integer variable for evaluation
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        if (solution.getVariable(i) instanceof BinaryVariable) {
                            sj.add(Boolean.toString(EncodingUtils.getBoolean(solution.getVariable(i))));
                        } else if ((solution.getVariable(i) instanceof RealVariable) && (!isEOSS)) {
                            sj.add(Double.toString(EncodingUtils.getReal(solution.getVariable(i))));
                        }
                    }
                }
                for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
                    sj.add(Double.toString((Double) solution.getAttribute(objectiveNames[i]))); // Normalized objective saved as an attribute of the solution
                }
                if (!isPartitioning) { // The constraint in the partitioning problem is purely due to the design formulation
                    for (int i = 0; i < solution.getNumberOfConstraints(); i++) {
                        sj.add(Double.toString(solution.getConstraint(i)));
                    }
                }
                for (String heuristicName : heuristicNames) {
                    sj.add(Double.toString((Double) solution.getAttribute(heuristicName)));
                }
                writer.append(sj.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveInternalPopulationOrArchive (String filename, Population population, String[] variableNames, String[] objectiveNames, String[] constraintNames, String[] heuristicNames, boolean isPartitioning, boolean isEOSS) {

        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving population or archive");

        try(FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            for (String variableName : variableNames) {
                headings.add(variableName);
            }
            for (String objectiveName : objectiveNames) {
                headings.add(objectiveName);
            }
            for (String constraintName : constraintNames) {
                headings.add(constraintName);
            }
            for (String heuristicName : heuristicNames) {
                headings.add(heuristicName);
            }
            writer.append(headings.toString());
            writer.append("\n");

            for (Solution solution : population) {
                StringJoiner sj = new StringJoiner(",");
                if (isPartitioning) { // Only for the Partitioning problem, extract integer variables and store
                    PartitioningArchitecture arch = (PartitioningArchitecture) solution;
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Integer.toString(((IntegerVariable)arch.getVariable(i)).getValue()));
                    }
                } else { // Other problems don't use integer variables and Assigning problem does not require the first integer variable for evaluation
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        if (solution.getVariable(i) instanceof BinaryVariable) {
                            sj.add(Boolean.toString(EncodingUtils.getBoolean(solution.getVariable(i))));
                        } else if ((solution.getVariable(i) instanceof RealVariable) && (!isEOSS)){
                            sj.add(Double.toString(EncodingUtils.getReal(solution.getVariable(i))));
                        }
                    }
                }

                for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
                    sj.add(Double.toString((Double) solution.getAttribute(objectiveNames[i])));
                }
                if (!isPartitioning) { // The constraint in the partitioning problem is purely due to the design formulation
                    for (int i = 0; i < solution.getNumberOfConstraints(); i++) {
                        sj.add(Double.toString(solution.getConstraint(i)));
                    }
                }
                for (String heuristicName : heuristicNames) {
                    sj.add(Double.toString((Double) solution.getAttribute(heuristicName)));
                }
                writer.append(sj.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
