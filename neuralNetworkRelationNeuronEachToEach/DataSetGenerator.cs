namespace neuralNetworkRelationNeuronEachToEach
{
    public class DataSetGenerator
    {

        static Random rnd = new Random();
        /// <summary>
        /// Метод getInputUniqueNumberDataRowArray() возвращяет
        /// массив с не повторяющимися числами
        /// </summary>
        /// <param name="count">количество чисел в массиве</param>
        /// <param name="maxVaalue">максимальное значение числа</param>
        public static double[] getInputUniqueNumberDataRowArray(int count, int maxVaalue = 10)
        {
            var input = new List<double>();
            for (int i = 0; i < count; i++)
            {
                var a = -1.0;
                bool rep = true;
                while (rep)
                {
                    a = rnd.Next(maxVaalue);
                    if (!input.Contains(a))
                    {
                        input.Add(a);
                        rep = false;
                    }
                }
            }
            return input.ToArray();
        }
        /// <summary>
        /// Метод getMaxId() возвращяет
        /// идентификатор максимального числа в массиве
        /// </summary>
        /// <param name="a">иследуемый массив</param>
        public static double getMaxId(double[] a)
        {
            int idMax = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] > a[idMax])
                {
                    idMax = i;
                }
            }
            return idMax + 1;
        }
        /// <summary>
        /// Метод getInputListNumber() возвращяет
        /// лист массивов из неповторяющихся чисел
        /// </summary>
        /// <param name="countDataRow">количество массивов в листе</param>
        /// <param name="countDataInOneRow">количество чисел в массиве</param>
        /// <param name="maxValueData">максимальное значение числа в массиве</param>
        public static List<double[]> getInputListNumber(int countDataRow, int countDataInOneRow, int maxValueData)
        {
            var list = new List<double[]>();
            for (int i = 0; i < countDataRow; i++)
            {
                list.Add(getInputUniqueNumberDataRowArray(countDataInOneRow, maxValueData));
            }
            return list;
        }
        /// <summary>
        /// Метод getOutputListForMaxId() возвращяет
        /// лист с идентификаторами максимальных чисел из input
        /// </summary>
        /// <param name="input">лист массивов с неповторяющимися значениями</param>
        public static List<double> getOutputListForMaxId(List<double[]> input)
        {
            var list = new List<double>();
            foreach (var item in input)
            {
                list.Add(getMaxId(item));
            }
            return list;
        }
        /// <summary>
        /// Метод getInputsMatrix() преобразует
        /// лист массивов в матрицу
        /// </summary>
        /// <param name="inputList">лист массивов с неповторяющимися</param>
        public static double[,] getInputsMatrix(List<double[]> inputList)
        {
            var input = new double[inputList.Count, inputList[0].Length];
            for (int i = 0; i < inputList.Count; i++)
            {
                for (int k = 0; k < inputList[i].Length; k++)
                {
                    input[i, k] = inputList[i][k];
                }
            }
            return input;
        }
        /// <summary>
        /// Метод showInputOutputData() выводит
        /// все строки матрици и ожидаемый результат к ней в строчку
        /// </summary>
        /// <param name="expected">массив ожидаемого результата</param>
        /// <param name="inputs">матрица входных данных</param>
        public static void showInputOutputData(double[] expected, double[,] inputs)
        {
            for (int i = 0; i < inputs.GetUpperBound(0) + 1; i++)
            {
                Console.Write($"expected - {expected[i]}\t");
                for (int k = 0; k < inputs.GetUpperBound(1) + 1; k++)
                {
                    Console.Write($"inputs - {inputs[i, k]},");
                }
                Console.WriteLine();
            }
        }
    }
}
