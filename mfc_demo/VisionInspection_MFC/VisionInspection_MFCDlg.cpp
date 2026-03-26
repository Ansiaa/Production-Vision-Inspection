
// VisionInspection_MFCDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "VisionInspection_MFC.h"
#include "VisionInspection_MFCDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CVisionInspectionMFCDlg 대화 상자



CVisionInspectionMFCDlg::CVisionInspectionMFCDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_VISIONINSPECTION_MFC_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CVisionInspectionMFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_EDIT1, m_editopenfile);
	DDX_Control(pDX, IDC_ORI, m_picUpload);
	DDX_Control(pDX, IDC_ORI2, m_picRoi);
	DDX_Control(pDX, IDC_ORI3, m_picRoiMask);
	DDX_Control(pDX, IDC_ORI4, m_picPreproc);
	DDX_Control(pDX, IDC_ORI5, m_picOverlay);
	DDX_Control(pDX, IDC_ORI6, m_picHeatmap);
	DDX_Control(pDX, IDC_LIST1, m_listResult);
	DDX_Control(pDX, IDC_COMBO1, m_cls);
	DDX_Control(pDX, IDC_LIST2, m_listSetting);
}

BEGIN_MESSAGE_MAP(CVisionInspectionMFCDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPEN, &CVisionInspectionMFCDlg::OnBnClickedOpen)
END_MESSAGE_MAP()


// CVisionInspectionMFCDlg 메시지 처리기

BOOL CVisionInspectionMFCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	m_cls.AddString(L"pcb4");
	m_cls.AddString(L"cashew");
	m_cls.SetCurSel(0);

	m_listResult.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
	m_listResult.InsertColumn(0, L"Item", LVCFMT_LEFT, 200);
	m_listResult.InsertColumn(1, L"Value", LVCFMT_LEFT, 250);

	m_listSetting.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
	m_listSetting.InsertColumn(0, L"Item", LVCFMT_LEFT, 200);
	m_listSetting.InsertColumn(1, L"Value", LVCFMT_LEFT, 250);

	PreparePictureControl(m_picUpload);
	PreparePictureControl(m_picRoi);
	PreparePictureControl(m_picRoiMask);
	PreparePictureControl(m_picPreproc);
	PreparePictureControl(m_picOverlay);
	PreparePictureControl(m_picHeatmap);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CVisionInspectionMFCDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CVisionInspectionMFCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CVisionInspectionMFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CVisionInspectionMFCDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	if (m_bServerStartedByApp && m_piServer.hProcess)
	{
		TerminateProcess(m_piServer.hProcess, 0);

		CloseHandle(m_piServer.hProcess);
		m_piServer.hProcess = nullptr;

		if (m_piServer.hThread)
		{
			CloseHandle(m_piServer.hThread);
			m_piServer.hThread = nullptr;
		}
	}
}
void CVisionInspectionMFCDlg::AddResultRow(int row, const CString& item, const CString& value, CListCtrl& m_list)
{
	m_list.InsertItem(row, item);
	m_list.SetItemText(row, 1, value);
}
//---------------------------------------------------------------------------------------------------------
bool CVisionInspectionMFCDlg::IsServerAlive()
{
	bool ok = false;

	HINTERNET hSession = WinHttpOpen(
		L"MFC FastAPI Client/1.0",
		WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
		WINHTTP_NO_PROXY_NAME,
		WINHTTP_NO_PROXY_BYPASS,
		0);

	if (!hSession)
		return false;

	HINTERNET hConnect = WinHttpConnect(
		hSession,
		L"127.0.0.1",
		8000,
		0);

	if (!hConnect)
	{
		WinHttpCloseHandle(hSession);
		return false;
	}

	HINTERNET hRequest = WinHttpOpenRequest(
		hConnect,
		L"GET",
		L"/health",
		nullptr,
		WINHTTP_NO_REFERER,
		WINHTTP_DEFAULT_ACCEPT_TYPES,
		0);

	if (!hRequest)
	{
		WinHttpCloseHandle(hConnect);
		WinHttpCloseHandle(hSession);
		return false;
	}

	BOOL bResults = WinHttpSendRequest(
		hRequest,
		WINHTTP_NO_ADDITIONAL_HEADERS,
		0,
		WINHTTP_NO_REQUEST_DATA,
		0,
		0,
		0);

	if (bResults)
		bResults = WinHttpReceiveResponse(hRequest, nullptr);

	if (bResults)
	{
		DWORD statusCode = 0;
		DWORD size = sizeof(statusCode);

		if (WinHttpQueryHeaders(
			hRequest,
			WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
			WINHTTP_HEADER_NAME_BY_INDEX,
			&statusCode,
			&size,
			WINHTTP_NO_HEADER_INDEX))
		{
			ok = (statusCode == 200);
		}
	}

	WinHttpCloseHandle(hRequest);
	WinHttpCloseHandle(hConnect);
	WinHttpCloseHandle(hSession);

	return ok;
}
bool CVisionInspectionMFCDlg::StartFastApiServer()
{
	if (m_piServer.hProcess || m_piServer.hThread)
		return true;

	CString projectRoot = L"D:\\project1";
	CString pythonExe = L"D:\\project1\\.venv\\Scripts\\python.exe";

	CString cmdLine;
	cmdLine.Format(
		L"\"%s\" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000",
		pythonExe);

	STARTUPINFO si{};
	si.cb = sizeof(si);

	ZeroMemory(&m_piServer, sizeof(m_piServer));

	// CreateProcess는 command line 버퍼 수정 가능해야 하므로 GetBuffer 사용
	wchar_t* pCmd = cmdLine.GetBuffer();

	BOOL created = CreateProcess(
		nullptr,
		pCmd,
		nullptr,
		nullptr,
		FALSE,
		CREATE_NO_WINDOW,
		nullptr,
		projectRoot,
		&si,
		&m_piServer);

	cmdLine.ReleaseBuffer();

	if (!created)
	{
		ZeroMemory(&m_piServer, sizeof(m_piServer));
		return false;
	}

	m_bServerStartedByApp = true;
	return true;
}

bool CVisionInspectionMFCDlg::EnsureServerRunning(DWORD waitMs)
{
	if (IsServerAlive())
		return true;

	if (!StartFastApiServer())
		return false;

	DWORD startTick = GetTickCount();

	while ((GetTickCount() - startTick) < waitMs)
	{
		if (IsServerAlive())
			return true;

		Sleep(300);
	}

	return false;
}
//---------------------------------------------------------------------------------------------------------
// 파일 전체를 binary로 읽는 함수
static bool ReadAllBytes(const CString& path, std::vector<BYTE>& out)
{
	CFile file;
	if (!file.Open(path, CFile::modeRead | CFile::shareDenyNone))
		return false;

	ULONGLONG fileSize = file.GetLength();
	out.resize((size_t)fileSize);

	if (fileSize > 0)
	{
		UINT readBytes = file.Read(out.data(), (UINT)fileSize);
		if (readBytes != fileSize)
		{
			file.Close();
			return false;
		}
	}

	file.Close();
	return true;
}

// 전체 경로에서 파일명만 추출
// 예: D:\img\000_02.JPG -> 000_02.JPG
static CString GetFileNameOnly(const CString& fullPath)
{
	int pos = fullPath.ReverseFind(L'\\');
	if (pos < 0)
		pos = fullPath.ReverseFind(L'/');

	if (pos >= 0)
		return fullPath.Mid(pos + 1);

	return fullPath;
}

// 확장자 기준으로 MIME 타입 결정
static CString GetMimeType(const CString& path)
{
	CString lower = path;
	lower.MakeLower();

	if (lower.Right(4) == L".jpg" || lower.Right(5) == L".jpeg")
		return L"image/jpeg";
	if (lower.Right(4) == L".png")
		return L"image/png";
	if (lower.Right(4) == L".bmp")
		return L"image/bmp";

	return L"application/octet-stream";
}

// CString(Wide) -> UTF-8 문자열 변환
static CStringA ToUtf8(const CString& text)
{
	int len = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);

	CStringA result;
	LPSTR buffer = result.GetBuffer(len);
	WideCharToMultiByte(CP_UTF8, 0, text, -1, buffer, len, nullptr, nullptr);
	result.ReleaseBuffer();

	return result;
}

// UTF-8 std::string -> CString 변환
static CString FromUtf8(const std::string& text)
{
	if (text.empty())
		return L"";

	int len = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), nullptr, 0);

	CString result;
	LPWSTR buffer = result.GetBuffer(len);
	MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), buffer, len);
	result.ReleaseBuffer(len);

	return result;
}

// JSON 문자열 값 추출
// 예: "cls":"pcb4" 에서 pcb4 추출
static CString GetJsonStringValue(const CString& json, const CString& key)
{
	CString pattern;
	pattern.Format(L"\"%s\":", key); // 따옴표를 여기서 바로 붙이지 않습니다.

	int start = json.Find(pattern);
	if (start < 0)
		return L"";

	start += pattern.GetLength();

	// 콜론(:) 뒤에 있는 공백(띄어쓰기, 줄바꿈 등)을 모두 무시하고 건너뜀
	while (start < json.GetLength() && iswspace(json[start]))
		start++;

	// 이제 시작하는 따옴표(")가 있어야 함
	if (start >= json.GetLength() || json[start] != L'\"')
		return L"";

	start++; // 시작 따옴표 다음 위치로 이동

	// 닫는 따옴표 찾기
	int end = json.Find(L"\"", start);
	if (end < 0)
		return L"";

	return json.Mid(start, end - start);
}

// JSON 숫자 값 추출
// 예: "pred_score":0.094 에서 0.094 추출
static CString GetJsonNumberValue(const CString& json, const CString& key)
{
	CString pattern;
	pattern.Format(L"\"%s\":", key);

	int start = json.Find(pattern);
	if (start < 0)
		return L"";

	start += pattern.GetLength();

	while (start < json.GetLength() && iswspace(json[start]))
		start++;

	int end = start;
	while (end < json.GetLength())
	{
		wchar_t ch = json[end];
		if ((ch >= L'0' && ch <= L'9') || ch == L'.' || ch == L'-')
			end++;
		else
			break;
	}

	return json.Mid(start, end - start);
}

// JSON bool 값 추출
// 예: "is_anomaly":true -> true
static bool GetJsonBoolValue(const CString& json, const CString& key, bool defaultValue = false)
{
	CString pattern;
	pattern.Format(L"\"%s\":", key);

	int start = json.Find(pattern);
	if (start < 0)
		return defaultValue;

	start += pattern.GetLength();

	while (start < json.GetLength() && iswspace(json[start]))
		start++;

	CString remain = json.Mid(start, 5);
	remain.MakeLower();

	if (remain.Left(4) == L"true")
		return true;
	if (remain.Left(5) == L"false")
		return false;

	return defaultValue;
}
bool CVisionInspectionMFCDlg::CallInspectApi(const CString& cls, const CString& filePath, CString& jsonResponse)
{
	jsonResponse.Empty();

	// 1) 업로드할 이미지 파일을 binary로 읽는다
	std::vector<BYTE> fileData;
	if (!ReadAllBytes(filePath, fileData))
		return false;

	// 2) multipart/form-data boundary 문자열
	CString boundary = L"----MFCBoundary7MA4YWxkTrZu0gW";

	// 3) 파일명과 MIME 타입 준비
	//    한글 경로를 따로 고려하지 않아도 된다고 했으니
	//    실제 파일명만 그대로 보낸다.
	CString fileName = GetFileNameOnly(filePath);
	CString mimeType = GetMimeType(filePath);

	// 4) multipart/form-data header/footer 구성
	CString partHeader;
	partHeader.Format(
		L"--%s\r\n"
		L"Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
		L"Content-Type: %s\r\n"
		L"\r\n",
		boundary, fileName, mimeType);

	CString partFooter;
	partFooter.Format(
		L"\r\n--%s--\r\n",
		boundary);

	CStringA partHeaderA = ToUtf8(partHeader);
	CStringA partFooterA = ToUtf8(partFooter);

	// 5) 실제 요청 body 생성
	//    [header][file binary][footer] 형태로 합친다
	std::vector<BYTE> requestBody;
	requestBody.insert(
		requestBody.end(),
		(BYTE*)(LPCSTR)partHeaderA,
		(BYTE*)(LPCSTR)partHeaderA + partHeaderA.GetLength());

	requestBody.insert(
		requestBody.end(),
		fileData.begin(),
		fileData.end());

	requestBody.insert(
		requestBody.end(),
		(BYTE*)(LPCSTR)partFooterA,
		(BYTE*)(LPCSTR)partFooterA + partFooterA.GetLength());

	// 6) 요청할 API 경로
	CString urlPath;
	urlPath.Format(L"/v1/inspect?cls=%s", cls);

	// 7) WinHTTP 세션 생성
	HINTERNET hSession = WinHttpOpen(
		L"MFC FastAPI Client/1.0",
		WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
		WINHTTP_NO_PROXY_NAME,
		WINHTTP_NO_PROXY_BYPASS,
		0);

	if (!hSession)
		return false;

	// 8) localhost:8000 연결
	HINTERNET hConnect = WinHttpConnect(
		hSession,
		L"127.0.0.1",
		8000,
		0);

	if (!hConnect)
	{
		WinHttpCloseHandle(hSession);
		return false;
	}

	// 9) POST 요청 생성
	HINTERNET hRequest = WinHttpOpenRequest(
		hConnect,
		L"POST",
		urlPath,
		nullptr,
		WINHTTP_NO_REFERER,
		WINHTTP_DEFAULT_ACCEPT_TYPES,
		0);

	if (!hRequest)
	{
		WinHttpCloseHandle(hConnect);
		WinHttpCloseHandle(hSession);
		return false;
	}

	// 10) HTTP 헤더 설정
	CString headers;
	headers.Format(
		L"Content-Type: multipart/form-data; boundary=%s\r\n"
		L"Accept: application/json\r\n",
		boundary);

	// 11) 요청 전송
	BOOL ok = WinHttpSendRequest(
		hRequest,
		headers,
		(DWORD)-1L,
		requestBody.data(),
		(DWORD)requestBody.size(),
		(DWORD)requestBody.size(),
		0);

	if (!ok)
	{
		WinHttpCloseHandle(hRequest);
		WinHttpCloseHandle(hConnect);
		WinHttpCloseHandle(hSession);
		return false;
	}

	// 12) 응답 수신
	ok = WinHttpReceiveResponse(hRequest, nullptr);
	if (!ok)
	{
		WinHttpCloseHandle(hRequest);
		WinHttpCloseHandle(hConnect);
		WinHttpCloseHandle(hSession);
		return false;
	}

	// 13) HTTP 상태 코드 확인
	DWORD statusCode = 0;
	DWORD statusCodeSize = sizeof(statusCode);

	WinHttpQueryHeaders(
		hRequest,
		WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
		WINHTTP_HEADER_NAME_BY_INDEX,
		&statusCode,
		&statusCodeSize,
		WINHTTP_NO_HEADER_INDEX);

	// 14) 응답 body(JSON) 읽기
	std::string responseUtf8;

	while (true)
	{
		DWORD availableSize = 0;
		if (!WinHttpQueryDataAvailable(hRequest, &availableSize))
			break;

		if (availableSize == 0)
			break;

		std::vector<char> buffer(availableSize);
		DWORD readSize = 0;

		if (!WinHttpReadData(hRequest, buffer.data(), availableSize, &readSize))
			break;

		responseUtf8.append(buffer.data(), readSize);
	}

	// 15) 핸들 정리
	WinHttpCloseHandle(hRequest);
	WinHttpCloseHandle(hConnect);
	WinHttpCloseHandle(hSession);

	// 16) UTF-8 JSON 응답을 CString으로 변환
	jsonResponse = FromUtf8(responseUtf8);

	// 17) HTTP 200이면 성공
	return (statusCode == 200);
}
bool CVisionInspectionMFCDlg::ParseInspectResult(const CString& jsonResponse, InspectResult& result)
{
	// 이전 값 초기화
	result = InspectResult{};

	// 문자열 값
	result.cls = GetJsonStringValue(jsonResponse, L"cls");
	result.gt_label = GetJsonStringValue(jsonResponse, L"gt_label");
	result.pred_label = GetJsonStringValue(jsonResponse, L"pred_label");
	result.decision_reason = GetJsonStringValue(jsonResponse, L"decision_reason");

	result.image_path = GetJsonStringValue(jsonResponse, L"image_path");
	result.roi_path = GetJsonStringValue(jsonResponse, L"roi_path");
	result.roi_mask_path = GetJsonStringValue(jsonResponse, L"roi_mask_path");
	result.preprocessed_path = GetJsonStringValue(jsonResponse, L"preprocessed_path");
	result.overlay_path = GetJsonStringValue(jsonResponse, L"overlay_path");
	result.heatmap_path = GetJsonStringValue(jsonResponse, L"heatmap_path");

	// bool 값
	result.is_correct = GetJsonBoolValue(jsonResponse, L"is_correct", false);

	// 숫자 값
	result.pred_score = GetJsonNumberValue(jsonResponse, L"pred_score");
	result.decision_thr = GetJsonNumberValue(jsonResponse, L"decision_thr");
	result.processing_ms = GetJsonNumberValue(jsonResponse, L"processing_ms");

	// 최소 필수값이 있으면 성공으로 본다
	if (result.cls.IsEmpty()) return false;
	if (result.pred_label.IsEmpty()) return false;
	if (result.overlay_path.IsEmpty()) return false;
	if (result.heatmap_path.IsEmpty()) return false;
	if (result.image_path.IsEmpty()) return false;

	return true;
}
//---------------------------------------------------------------------------------------------------------
void CVisionInspectionMFCDlg::init()
{
	m_editopenfile.SetWindowTextW(_T(""));
	m_listResult.DeleteAllItems();
	m_listSetting.DeleteAllItems();

	auto ClearPic = [](CStatic& pic,HBITMAP& hBmp)
	{
		pic.SetBitmap(nullptr);
		if (hBmp)
		{
			DeleteObject(hBmp);
			hBmp = nullptr;
		}
	};

	ClearPic(m_picUpload, m_hBmpUpload);
	ClearPic(m_picRoi, m_hBmpRoi);
	ClearPic(m_picRoiMask, m_hBmpRoiMask);
	ClearPic(m_picPreproc, m_hBmpPreproc);
	ClearPic(m_picOverlay, m_hBmpOverlay);
	ClearPic(m_picHeatmap, m_hBmpHeatmap);

}
void CVisionInspectionMFCDlg::PreparePictureControl(CStatic& pic)
{
	HWND hWnd = pic.GetSafeHwnd();
	if (!hWnd)
		return;

	// 현재 style 읽기
	LONG_PTR style = ::GetWindowLongPtr(hWnd, GWL_STYLE);

	// static control의 "타입 비트" 전체 제거
	style &= ~SS_TYPEMASK;

	// bitmap 표시용으로 강제 변경
	style |= SS_BITMAP | WS_CHILD | WS_VISIBLE;

	::SetWindowLongPtr(hWnd, GWL_STYLE, style);

	// 다시 그리기 반영
	::SetWindowPos(
		hWnd,
		nullptr,
		0, 0, 0, 0,
		SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
	);

	pic.SetWindowTextW(L"");
	pic.Invalidate();
	pic.UpdateWindow();
}
bool CVisionInspectionMFCDlg::LoadImageToPicture(const CString& imagePath, CStatic& pictureCtrl, HBITMAP& hBitmap)
{
	if (imagePath.IsEmpty())
		return false;

	// 이전 bitmap 제거
	if (hBitmap != nullptr)
	{
		::SendMessage(pictureCtrl.GetSafeHwnd(), STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)nullptr);
		::DeleteObject(hBitmap);
		hBitmap = nullptr;
	}

	CImage image;
	HRESULT hr = image.Load(imagePath);
	if (FAILED(hr))
	{
		CString msg;
		msg.Format(L"이미지 로드 실패:\n%s", imagePath);
		AfxMessageBox(msg);
		return false;
	}

	CRect rc;
	pictureCtrl.GetClientRect(&rc);

	int ctrlW = rc.Width();
	int ctrlH = rc.Height();
	int imgW = image.GetWidth();
	int imgH = image.GetHeight();

	if (ctrlW <= 0 || ctrlH <= 0 || imgW <= 0 || imgH <= 0)
	{
		image.Destroy();
		return false;
	}

	// 비율 유지
	double scaleX = (double)ctrlW / (double)imgW;
	double scaleY = (double)ctrlH / (double)imgH;
	double scale = min(scaleX, scaleY);

	int drawW = (int)(imgW * scale);
	int drawH = (int)(imgH * scale);

	int offsetX = (ctrlW - drawW) / 2;
	int offsetY = (ctrlH - drawH) / 2;

	CImage canvas;
	canvas.Create(ctrlW, ctrlH, 24);

	HDC hDC = canvas.GetDC();

	::SetStretchBltMode(hDC, HALFTONE);
	::SetBrushOrgEx(hDC, 0, 0, nullptr);

	RECT fillRect = { 0, 0, ctrlW, ctrlH };
	::FillRect(hDC, &fillRect, (HBRUSH)::GetStockObject(WHITE_BRUSH));

	image.Draw(hDC, offsetX, offsetY, drawW, drawH);

	canvas.ReleaseDC();

	hBitmap = canvas.Detach();

	// SetBitmap 대신 직접 메시지 전송
	::SendMessage(pictureCtrl.GetSafeHwnd(), STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hBitmap);

	pictureCtrl.Invalidate();
	pictureCtrl.UpdateWindow();

	image.Destroy();
	return true;
}
void CVisionInspectionMFCDlg::LoadSettingsFromJson(const CString& targetCls)
{
	CString filePath = L"D:\\project1\\config\\thresholds_final_operating.json";

	std::vector<BYTE> fileData;
	if (!ReadAllBytes(filePath, fileData)) {
		// 파일을 찾을 수 없거나 읽기 실패 시
		return;
	}

	// 바이트 데이터를 UTF-8 문자열로 변환 후 CString으로 변환
	std::string utf8Str((char*)fileData.data(), fileData.size());
	CString jsonStr = FromUtf8(utf8Str);

	// 선택된 클래스(예: "cashew":)의 블록 위치 찾기
	CString pattern;
	pattern.Format(L"\"%s\":", targetCls);
	int startPos = jsonStr.Find(pattern);
	if (startPos < 0) return;

	// 해당 클래스의 { } 블록 영역만 추출
	int blockStart = jsonStr.Find(L"{", startPos);
	int blockEnd = jsonStr.Find(L"}", startPos);
	if (blockStart < 0 || blockEnd < 0) return;

	CString blockJson = jsonStr.Mid(blockStart, blockEnd - blockStart + 1);

	// 블록 내에서 threshold와 variant 값 추출
	CString threshold = GetJsonNumberValue(blockJson, L"threshold");
	CString variant = GetJsonStringValue(blockJson, L"variant");

	int row = 0;
	AddResultRow(row++, L"threshold", threshold, m_listSetting);
	AddResultRow(row++, L"variant", variant, m_listSetting);
}
void CVisionInspectionMFCDlg::OnBnClickedOpen()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	if (!EnsureServerRunning())
	{
		AfxMessageBox(L"FastAPI 서버를 시작하지 못했습니다.");
		return;
	}

	init();

	int nSelected = m_cls.GetCurSel();
	if (nSelected < 0)
	{
		AfxMessageBox(L"검사할 클래스를 선택하세요.");
		return;
	}
	CString cls;
	m_cls.GetLBText(nSelected, cls);
	LoadSettingsFromJson(cls);
	CFileDialog dlg(TRUE, nullptr, nullptr,
		OFN_FILEMUSTEXIST,
		L"Image Files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp||");

	if (dlg.DoModal() != IDOK)
		return;

	m_selectfilepath = dlg.GetPathName();
	m_editopenfile.SetWindowTextW(m_selectfilepath);


	CString jsonResponse;
	if (!CallInspectApi(cls, m_selectfilepath, jsonResponse))
	{
		AfxMessageBox(L"API 호출 실패");
		return;
	}

	InspectResult result;
	if (!ParseInspectResult(jsonResponse, result))
	{
		AfxMessageBox(L"JSON 파싱 실패");
		return;
	}
	auto NormalizePath = [](CString& path)
		{
			path.Replace(L"/", L"\\");
		};

	NormalizePath(result.image_path);
	NormalizePath(result.roi_path);
	NormalizePath(result.roi_mask_path);
	NormalizePath(result.preprocessed_path);
	NormalizePath(result.overlay_path);
	NormalizePath(result.heatmap_path);

	LoadImageToPicture(result.image_path, m_picUpload, m_hBmpUpload);
	LoadImageToPicture(result.roi_path, m_picRoi, m_hBmpRoi);
	LoadImageToPicture(result.roi_mask_path, m_picRoiMask, m_hBmpRoiMask);
	LoadImageToPicture(result.preprocessed_path, m_picPreproc, m_hBmpPreproc);
	LoadImageToPicture(result.overlay_path, m_picOverlay, m_hBmpOverlay);
	LoadImageToPicture(result.heatmap_path, m_picHeatmap, m_hBmpHeatmap);

	int row = 0;
	AddResultRow(row++, L"cls", cls,m_listResult);
	AddResultRow(row++, L"gt_label", result.gt_label, m_listResult);
	AddResultRow(row++, L"pred_label", result.pred_label, m_listResult);
	AddResultRow(row++, L"is_correct", result.is_correct ? L"true" : L"false", m_listResult);
	AddResultRow(row++, L"pred_score", result.pred_score, m_listResult);
	AddResultRow(row++, L"decision_thr", result.decision_thr, m_listResult);
	AddResultRow(row++, L"processing_ms", result.processing_ms, m_listResult);


}
