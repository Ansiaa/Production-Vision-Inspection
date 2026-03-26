
// VisionInspection_MFCDlg.h: 헤더 파일
//

#pragma once
#include <winhttp.h>
#include <vector>
#include <string>
#include <atlstr.h>

#pragma comment(lib, "winhttp.lib")

struct InspectResult
{
	CString cls;
	CString gt_label;
	CString pred_label;
	bool    is_correct = false;

	CString pred_score;
	CString decision_thr;
	CString decision_reason;
	CString processing_ms;

	CString image_path;
	CString roi_path;
	CString roi_mask_path;
	CString preprocessed_path;
	CString overlay_path;
	CString heatmap_path;
};

// CVisionInspectionMFCDlg 대화 상자
class CVisionInspectionMFCDlg : public CDialogEx
{
// 생성입니다.
public:
	CVisionInspectionMFCDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_VISIONINSPECTION_MFC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

protected:
	PROCESS_INFORMATION m_piServer{};
	bool m_bServerStartedByApp = false;

	virtual void OnDestroy();
	void AddResultRow(int row, const CString& item, const CString& value, CListCtrl& m_list);

public:
	// FastAPI 서버 관련
	bool IsServerAlive();
	bool StartFastApiServer();
	bool EnsureServerRunning(DWORD waitMs = 15000);

	bool CallInspectApi(const CString& cls, const CString& filePath, CString& jsonResponse);
	bool ParseInspectResult(const CString& jsonResponse, InspectResult& result);

	void init();
	void PreparePictureControl(CStatic& pic);
	bool LoadImageToPicture(const CString& imagePath, CStatic& pictureCtrl, HBITMAP& hBitmap);
	void LoadSettingsFromJson(const CString& targetCls);

	afx_msg void OnBnClickedOpen();
    CListCtrl m_listResult;
	CEdit m_editopenfile;
	CComboBox m_cls;
	CListCtrl m_listSetting;

	CString m_selectfilepath;

	CStatic m_picUpload;
	CStatic m_picRoi;
	CStatic m_picRoiMask;
	CStatic m_picPreproc;
	CStatic m_picOverlay;
	CStatic m_picHeatmap;

	HBITMAP m_hBmpUpload = nullptr;
	HBITMAP m_hBmpRoi = nullptr;
	HBITMAP m_hBmpRoiMask = nullptr;
	HBITMAP m_hBmpPreproc = nullptr;
	HBITMAP m_hBmpOverlay = nullptr;
	HBITMAP m_hBmpHeatmap = nullptr;
	
	
};
