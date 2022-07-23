using System.Drawing;
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
        public static double[] generateInputUniqueNumberDataRowArray(int count, int maxValue = 10)
        {
            var input = new List<double>();
            for (int i = 0; i < count; i++)
            {
                var a = -1.0;
                bool rep = true;
                while (rep)
                {
                    a = Convert.ToDouble("0," + rnd.Next(maxValue).ToString());
                    //a = rnd.Next(10);
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
            return idMax;
        }

        public static double[,] createInputMatrixForMaxId(ref double[] expected, int rowCount, int collumCount, int maxValue = 10)
        {
            double[] exp = new double[rowCount];
            var matrix = new double[rowCount, collumCount];
            for (int i = 0; i < rowCount; i++)
            {
                var input = generateInputUniqueNumberDataRowArray(collumCount, maxValue);
                exp[i] = getMaxId(input);
                for (int k = 0; k < collumCount; k++)
                {
                    matrix[i, k] = input[k];
                }
            }
            expected = exp;
            return matrix;
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
                    Console.Write($"inputs - {inputs[i, k]}\t");
                }
                Console.WriteLine();
            }
        }

        public static double[] ImageToArray(string path)
        {
            Bitmap bitmap = new Bitmap(path);
            List<double> list = new List<double>();
            for (int i = 0; i < bitmap.Size.Width; i++)
            {
                for (int k = 0; k < bitmap.Size.Height; k++)
                {
                    list.Add(bitmap.GetPixel(i, k).R / 255);
                }
            }
            return list.ToArray();
        }
        public static double[,] createInputMatrixForImage(string dataSetFolderPath, int c, ref double[] expected)
        {

            string[] allfiles = Directory.GetFiles(dataSetFolderPath);
            int countRow = c;
            var imgSizeFile = new Bitmap(allfiles[0]).Size;
            double[] exp = new double[countRow];
            var matrix = new double[countRow, imgSizeFile.Width * imgSizeFile.Height];
            //allfiles.Length=60 000
            for (int i = 0; i < countRow; i++)
            {
                var imgArray = ImageToArray(allfiles[i]);
                for (int k = 0; k < imgSizeFile.Width * imgSizeFile.Height; k++)
                {
                    matrix[i, k] = imgArray[k];
                }
                exp[i] = Convert.ToDouble(allfiles[i][allfiles[i].LastIndexOf("-num") + 4].ToString());
            }
            expected = exp;
            return matrix;
        }
        public static double[,] createInputMatrixForAutoEncoder(ref double[,] expected, int rowCount, int collumCount, int maxValue = 10)
        {
            var matrix = new double[rowCount, collumCount];
            for (int i = 0; i < rowCount; i++)
            {
                var data = generateInputUniqueNumberDataRowArray(expected.GetLength(1), maxValue);
                for (int k = 0; k < collumCount; k++)
                {
                    matrix[i, k] = data[k];
                }
            }
            expected = matrix;
            return matrix;
        }
    }
}
