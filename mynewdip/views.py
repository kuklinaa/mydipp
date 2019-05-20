from django.shortcuts import render_to_response
import joblib
from django.template.context_processors import csrf
import numpy as np
import math

# Create your views here.


def appropriate_matrix_crm(marks):
	matrix = np.zeros(shape=(10, 10))
	print(matrix.shape)
	for first_index, first_value in enumerate(marks):
		for second_index, second_value in enumerate(marks):
			if (first_value - second_value) % 2 == 0:
				difference = abs(first_value - second_value) + 1
			else:
				difference = abs(first_value - second_value)
			if first_value - second_value < 0:
				matrix[first_index][second_index] = 1 / difference
			elif first_value - second_value == 0:
				matrix[first_index][second_index] = 1
			else:
				matrix[first_index][second_index] = difference

	results = []
	for row in matrix:
		results.append(math.sqrt(row.prod()))
	_sum = sum(results)
	final_result = (np.array(results)) / _sum
	return final_result


def experts_marks(experts_marks):
	total_results = []
	for marks in experts_marks:
		# marks = [math.ceil(x) for x in marks]
		matrix = np.zeros(shape=(5, 5))
		for first_index, first_value in enumerate(marks):
			for second_index, second_value in enumerate(marks):
				if (first_value - second_value) % 2 == 0:
					difference = abs(first_value - second_value) + 1
				else:
					difference = abs(first_value - second_value)
				if first_value - second_value < 0:
					matrix[first_index][second_index] = 1 / difference
				elif first_value - second_value == 0:
					matrix[first_index][second_index] = 1
				else:
					matrix[first_index][second_index] = difference

		results = []
		for row in matrix:
			row_p = 1
			for row_elem in row:
				row_p *= row_elem
			results.append(math.sqrt(row_p))
		_sum = sum(results)
		final_result = (np.array(results)) / _sum
		total_results.append(final_result)
	print(total_results)
	return total_results


def get_key_by_value(dictionary, value):
	for key, item in dictionary.items():
		if item == value:
			return key
	return None


def calculate_crms_weights(user_coefs, experts_marks):
	print("user coef", user_coefs)
	print("exp marks", experts_marks)
	results = []
	user_coefs = np.array(user_coefs)
	for column in range(5):
		vector = np.array([row[column] for row in experts_marks])
		results.append(user_coefs * vector)
	sums = [sum(row) for row in results]
	return sums


def answer_getter(request):

	expert_marks_meanings = [[4, 6, 3, 7, 8],[4, 6, 7, 6, 6],[2, 3, 6, 7, 6],[8, 7, 3, 3, 9],[9, 9, 6, 6, 2],[4, 9, 6, 9, 2],[9, 2, 6, 4, 6],[4, 3, 10, 4, 7],[4, 8, 9, 4, 4],[1, 6, 2, 6, 2]]


	args = {}
	args.update(csrf(request))
	if request.POST:
		systems = {'amoCRM': 1, 'Bitrix24': 2, 'MegaPlan': 3, 'RetailCRM': 4, 'FreshOffice': 5}
		path = "static/"
		knn_model = joblib.load(path + "knn.pkl")
		svm_model = joblib.load(path + "svm.pkl")
		dt_model = joblib.load(path + "dt.pkl")
		results = []
		marks = []
		for i in range(1,11):
			marks.append(int(request.POST.get("mark_" + str(i))))

		sums = calculate_crms_weights(appropriate_matrix_crm(marks), experts_marks(expert_marks_meanings))
		print(sums)
		marks = np.reshape(marks, (1, -1))

		CRMS = ['amoCRM', 'Bitrix24', 'MegaPlan', 'RetailCRM', 'FreshOffice']
		arm_coefs = dict(zip(CRMS, sums))
		sums.sort(reverse=True)
		print(arm_coefs)
		print(sums)

		knn_result = knn_model.predict(marks)
		knn_crm = get_key_by_value(dictionary=systems, value=knn_result[0])
		results.append(knn_crm)
		print(knn_crm)


		svm_result = svm_model.predict(marks)
		svm_crm = get_key_by_value(dictionary=systems, value=svm_result[0])
		results.append(svm_crm)
		print(svm_crm)


		dt_result = dt_model.predict(marks)
		dt_crm = get_key_by_value(dictionary=systems, value=dt_result[0])
		results.append(dt_crm)
		print(dt_crm)

		mat_crm = get_key_by_value(dictionary=arm_coefs, value=sums[0])
		results.append(mat_crm)
		print(mat_crm)

		mat_crm2 = get_key_by_value(dictionary=arm_coefs, value=sums[1])
		results.append(mat_crm)
		print(mat_crm2)


		majority = max(results, key=results.count)
		args.update({"CRM":majority})
		return render_to_response("result.html", args)

	else:
		return render_to_response("home.html", args)


